package main

import (
	"flag"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseOptions(t *testing.T) {
	opts, err := parseOptions([]string{"--player1-weights=exported.onnx", "--onnxruntime=/tmp/runtime"}, io.Discard)
	require.NoError(t, err)
	require.Equal(t, "exported.onnx", opts.weights)
	require.Equal(t, "/tmp/runtime", opts.runtime)
	opts, err = parseOptions([]string{"--player1-weights=exported.onnx"}, io.Discard)
	require.NoError(t, err)
	require.Empty(t, opts.runtime, "empty path delegates environment fallback to inference")
	for _, args := range [][]string{nil, {"--unknown"}, {"--player1-weights=x", "extra"}} {
		_, err := parseOptions(args, io.Discard)
		require.Error(t, err)
	}
	_, err = parseOptions([]string{"--help"}, io.Discard)
	require.ErrorIs(t, err, flag.ErrHelp)
}
