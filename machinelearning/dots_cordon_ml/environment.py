"""A minimal synchronous client for the stateful Dots Cordon game server."""

from __future__ import annotations

from dataclasses import dataclass

import grpc

from .proto import game_pb2, game_pb2_grpc


@dataclass(frozen=True, slots=True)
class StepResult:
    game: game_pb2.GameState
    player: int
    reward: float


class GameEnvironment:
    """Own one server-side game and expose reset/step operations."""

    def __init__(
        self,
        target: str,
        rows: int,
        columns: int,
        max_turns: int = 0,
        rpc_timeout: float = 10.0,
        connect_timeout: float = 10.0,
    ) -> None:
        if not 1 <= rows <= 255 or not 1 <= columns <= 255:
            raise ValueError("rows and columns must be between 1 and 255")
        if max_turns < 0:
            raise ValueError("max_turns must be non-negative")

        self.rows = rows
        self.columns = columns
        self.max_turns = max_turns
        self.rpc_timeout = rpc_timeout
        self._channel = grpc.insecure_channel(target)
        try:
            grpc.channel_ready_future(self._channel).result(timeout=connect_timeout)
        except grpc.FutureTimeoutError as exc:
            self._channel.close()
            raise ConnectionError(
                f"Dots Cordon server at {target} was not ready within "
                f"{connect_timeout:g}s"
            ) from exc

        self._stub = game_pb2_grpc.GameServiceStub(self._channel)
        try:
            response = self._stub.CreateGame(
                game_pb2.CreateGameRequest(
                    rows=rows,
                    columns=columns,
                    max_turns=max_turns,
                ),
                timeout=self.rpc_timeout,
            )
        except grpc.RpcError:
            self._channel.close()
            raise
        self.game = response.game
        self._closed = False

    def reset(self) -> game_pb2.GameState:
        self._ensure_open()
        response = self._stub.ResetGame(
            game_pb2.ResetGameRequest(game_id=self.game.game_id),
            timeout=self.rpc_timeout,
        )
        self.game = response.game
        return self.game

    def step(self, action: int) -> StepResult:
        self._ensure_open()
        board_size = self.rows * self.columns
        if not 0 <= action < board_size:
            raise ValueError(f"action must be in [0, {board_size}), got {action}")
        if self.game.terminal:
            raise RuntimeError("cannot step a terminal game; call reset first")
        if self.game.board.cells[action] != game_pb2.CELL_EMPTY:
            raise ValueError(f"action {action} selects a non-empty cell")

        player = self.game.current_player
        row, column = divmod(action, self.columns)
        response = self._stub.MakeMove(
            game_pb2.MakeMoveRequest(
                game_id=self.game.game_id,
                expected_turn=self.game.turn,
                position=game_pb2.Coordinate(row=row, column=column),
            ),
            timeout=self.rpc_timeout,
        )
        self.game = response.game
        return StepResult(
            game=self.game,
            player=player,
            reward=float(response.result.scored_points),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stub.DeleteGame(
                game_pb2.DeleteGameRequest(game_id=self.game.game_id),
                timeout=self.rpc_timeout,
            )
        except grpc.RpcError:
            # The server may already have gone away during shutdown.
            pass
        finally:
            self._closed = True
            self._channel.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")

    def __enter__(self) -> GameEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
