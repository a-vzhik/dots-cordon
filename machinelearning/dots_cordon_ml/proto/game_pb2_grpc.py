"""Client bindings for dotscordon.v1.GameService.

The protobuf messages are generated from api/dotscordon/v1/game.proto. This
small client-only stub is kept explicit because the trainer never hosts the
service itself.
"""

import grpc

from . import game_pb2


class GameServiceStub:
    def __init__(self, channel: grpc.Channel) -> None:
        self.SimulateMove = channel.unary_unary(
            "/dotscordon.v1.GameService/SimulateMove",
            request_serializer=game_pb2.SimulateMoveRequest.SerializeToString,
            response_deserializer=game_pb2.MakeMoveResponse.FromString,
        )
        self.CreateGame = channel.unary_unary(
            "/dotscordon.v1.GameService/CreateGame",
            request_serializer=game_pb2.CreateGameRequest.SerializeToString,
            response_deserializer=game_pb2.CreateGameResponse.FromString,
        )
        self.GetGame = channel.unary_unary(
            "/dotscordon.v1.GameService/GetGame",
            request_serializer=game_pb2.GetGameRequest.SerializeToString,
            response_deserializer=game_pb2.GetGameResponse.FromString,
        )
        self.MakeMove = channel.unary_unary(
            "/dotscordon.v1.GameService/MakeMove",
            request_serializer=game_pb2.MakeMoveRequest.SerializeToString,
            response_deserializer=game_pb2.MakeMoveResponse.FromString,
        )
        self.ResetGame = channel.unary_unary(
            "/dotscordon.v1.GameService/ResetGame",
            request_serializer=game_pb2.ResetGameRequest.SerializeToString,
            response_deserializer=game_pb2.ResetGameResponse.FromString,
        )
        self.DeleteGame = channel.unary_unary(
            "/dotscordon.v1.GameService/DeleteGame",
            request_serializer=game_pb2.DeleteGameRequest.SerializeToString,
            response_deserializer=game_pb2.DeleteGameResponse.FromString,
        )
