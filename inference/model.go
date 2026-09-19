// Package inference loads exported Dots Cordon policies and runs them in process.
// It uses ONNX Runtime's native library; Python is only needed when exporting.
package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Metadata struct {
	Version  int    `json:"version"`
	Encoding string `json:"encoding"`
	// Rows and Columns describe the training board. Version 2 exports accept
	// any supported board dimensions at inference time.
	Rows    int    `json:"rows"`
	Columns int    `json:"columns"`
	Episode int    `json:"episode"`
	Kind    string `json:"kind"`
}

// ONNX Runtime has one process-wide environment. Keep it alive until the last
// model closes, including when two models play against each other.
var environment struct {
	sync.Mutex
	users int
	path  string
}

func acquireEnvironment(path string) error {
	environment.Lock()
	defer environment.Unlock()
	if path == "" {
		path = os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	}
	if environment.users > 0 {
		if path != environment.path {
			return errors.New("loaded models must use the same ONNX Runtime library")
		}
	} else {
		ort.SetSharedLibraryPath(path)
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("load ONNX Runtime (set --onnxruntime or ONNXRUNTIME_SHARED_LIBRARY_PATH): %w", err)
		}
		environment.path = path
	}
	environment.users++
	return nil
}

func releaseEnvironment() {
	environment.Lock()
	defer environment.Unlock()
	environment.users--
	if environment.users == 0 {
		_ = ort.DestroyEnvironment()
	}
}

type Model struct {
	info    Metadata
	session *ort.DynamicAdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
	rows    int
	columns int
	mu      sync.Mutex
	closed  bool
}

// LoadWeights reads an ONNX policy exported with dots-cordon-export-policy.
// libraryPath may be empty to use ONNXRUNTIME_SHARED_LIBRARY_PATH.
func LoadWeights(path, libraryPath string) (_ *Model, err error) {
	if strings.HasPrefix(path, "champion:") || strings.HasPrefix(path, "checkpoint:") || strings.HasSuffix(path, ".pt") {
		return nil, errors.New("export this checkpoint first with dots-cordon-export-policy; Go loads the resulting .onnx file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if err = acquireEnvironment(libraryPath); err != nil {
		return nil, err
	}
	m := &Model{}
	defer func() {
		if err != nil {
			m.Close()
		}
	}()
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()
	if err = options.SetIntraOpNumThreads(1); err != nil {
		return nil, err
	}
	m.session, err = ort.NewDynamicAdvancedSessionWithONNXData(data, []string{"state"}, []string{"scores"}, options)
	if err != nil {
		return nil, err
	}
	metadata, err := m.session.GetModelMetadata()
	if err != nil {
		return nil, err
	}
	defer metadata.Destroy()
	value, found, err := metadata.LookupCustomMetadataMap("dots_cordon")
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, errors.New("missing Dots Cordon export metadata")
	}
	if err = json.Unmarshal([]byte(value), &m.info); err != nil {
		return nil, err
	}
	info := m.info
	if (info.Version != 1 && info.Version != 2) || info.Encoding != "player-relative-5-v1" ||
		info.Rows < 1 || info.Rows > 255 || info.Columns < 1 || info.Columns > 255 ||
		(info.Kind != "dqn" && info.Kind != "policy_value") {
		return nil, errors.New("unsupported Dots Cordon model metadata")
	}
	if err = m.resize(info.Rows, info.Columns); err != nil {
		return nil, err
	}
	// Validate tensor names, types and dimensions before a game starts.
	if err = m.session.Run([]ort.Value{m.input}, []ort.Value{m.output}); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *Model) Metadata() Metadata { return m.info }

// ValidateBoardSize checks game dimensions without changing the loaded model.
// Old, fixed-size exports remain usable on their original board.
func (m *Model) ValidateBoardSize(rows, columns int) error {
	if rows < 1 || rows > 255 || columns < 1 || columns > 255 {
		return errors.New("board dimensions must be between 1 and 255")
	}
	if m.info.Version == 1 && (rows != m.info.Rows || columns != m.info.Columns) {
		return fmt.Errorf("this ONNX export is fixed to %dx%d; re-export the checkpoint with dots-cordon-export-policy to support other board sizes", m.info.Rows, m.info.Columns)
	}
	return nil
}

// resize must be called with exclusive access to the model. Compare dimensions,
// not cell count: a 5x15 board needs a different tensor shape from a 15x5 board.
func (m *Model) resize(rows, columns int) error {
	if rows == m.rows && columns == m.columns {
		return nil
	}
	input, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 5, int64(rows), int64(columns)))
	if err != nil {
		return err
	}
	output, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(rows*columns)))
	if err != nil {
		_ = input.Destroy()
		return err
	}
	if m.input != nil {
		_ = m.input.Destroy()
	}
	if m.output != nil {
		_ = m.output.Destroy()
	}
	m.input, m.output = input, output
	m.rows, m.columns = rows, columns
	return nil
}

// Infer returns one raw action score per cell in row-major order. Callers mask
// illegal actions and take the first maximum, matching greedy PyTorch inference.
// The same loaded model can be used with different board dimensions.
func (m *Model) Infer(state []float32, rows, columns int) ([]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, errors.New("model is closed")
	}
	if err := m.ValidateBoardSize(rows, columns); err != nil {
		return nil, err
	}
	if len(state) != 5*rows*columns {
		return nil, errors.New("incorrect encoded state size")
	}
	if err := m.resize(rows, columns); err != nil {
		return nil, err
	}
	copy(m.input.GetData(), state)
	if err := m.session.Run([]ort.Value{m.input}, []ort.Value{m.output}); err != nil {
		return nil, err
	}
	return append([]float32(nil), m.output.GetData()...), nil
}

func (m *Model) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return
	}
	m.closed = true
	if m.session != nil {
		_ = m.session.Destroy()
	}
	if m.input != nil {
		_ = m.input.Destroy()
	}
	if m.output != nil {
		_ = m.output.Destroy()
	}
	releaseEnvironment()
}
