package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/opponent"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/session"
	"github.com/a-vzhik/dots-cordon/runners/ebiten/internal/view"
	"github.com/hajimehoshi/ebiten/v2"
)

type options struct{ weights, runtime string }

func parseOptions(args []string, output io.Writer) (options, error) {
	var opts options
	flags := flag.NewFlagSet("dots-cordon", flag.ContinueOnError)
	flags.SetOutput(output)
	flags.StringVar(&opts.weights, "player1-weights", "", "required exported ONNX model for player 1")
	flags.StringVar(&opts.runtime, "onnxruntime", "", "ONNX Runtime shared library (defaults to ONNXRUNTIME_SHARED_LIBRARY_PATH)")
	if err := flags.Parse(args); err != nil {
		return opts, err
	}
	if flags.NArg() != 0 {
		return opts, errors.New("unexpected positional arguments")
	}
	if opts.weights == "" {
		return opts, errors.New("--player1-weights=exported.onnx is required")
	}
	return opts, nil
}

func run(args []string) error {
	opts, err := parseOptions(args, os.Stderr)
	if err != nil {
		return err
	}
	model, err := opponent.Load(opts.weights, opts.runtime)
	if err != nil {
		return fmt.Errorf("load player 1: %w", err)
	}
	defer model.Close()
	controller := session.NewController(model)
	defer controller.Close() // Join inference before destroying native resources.
	ebiten.SetWindowSize(view.Width, view.Height)
	ebiten.SetWindowTitle("Dots Cordon")
	if err := ebiten.RunGame(view.New(controller)); err != nil {
		return err
	}
	return controller.Err()
}

func main() {
	// RunGame must remain on the main goroutine.
	if err := run(os.Args[1:]); err != nil && !errors.Is(err, flag.ErrHelp) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
