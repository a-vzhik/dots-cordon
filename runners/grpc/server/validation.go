package grpcserver

import (
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func validateDimensions(rows, columns uint32) error {
	if rows == 0 || rows > 255 {
		return status.Error(codes.InvalidArgument, "rows must be between 1 and 255")
	}
	if columns == 0 || columns > 255 {
		return status.Error(codes.InvalidArgument, "columns must be between 1 and 255")
	}
	return nil
}
