from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Cell(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CELL_EMPTY: _ClassVar[Cell]
    CELL_PLAYER_0: _ClassVar[Cell]
    CELL_PLAYER_1: _ClassVar[Cell]
    CELL_DEAD_EMPTY: _ClassVar[Cell]
    CELL_DEAD_PLAYER_0: _ClassVar[Cell]
    CELL_DEAD_PLAYER_1: _ClassVar[Cell]

class TerminationReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TERMINATION_REASON_UNSPECIFIED: _ClassVar[TerminationReason]
    TERMINATION_REASON_BOARD_FULL: _ClassVar[TerminationReason]
    TERMINATION_REASON_TURN_LIMIT: _ClassVar[TerminationReason]
CELL_EMPTY: Cell
CELL_PLAYER_0: Cell
CELL_PLAYER_1: Cell
CELL_DEAD_EMPTY: Cell
CELL_DEAD_PLAYER_0: Cell
CELL_DEAD_PLAYER_1: Cell
TERMINATION_REASON_UNSPECIFIED: TerminationReason
TERMINATION_REASON_BOARD_FULL: TerminationReason
TERMINATION_REASON_TURN_LIMIT: TerminationReason

class SimulateMoveRequest(_message.Message):
    __slots__ = ("game", "position", "max_turns")
    GAME_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    MAX_TURNS_FIELD_NUMBER: _ClassVar[int]
    game: GameState
    position: Coordinate
    max_turns: int
    def __init__(self, game: _Optional[_Union[GameState, _Mapping]] = ..., position: _Optional[_Union[Coordinate, _Mapping]] = ..., max_turns: _Optional[int] = ...) -> None: ...

class CreateGameRequest(_message.Message):
    __slots__ = ("rows", "columns", "max_turns")
    ROWS_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    MAX_TURNS_FIELD_NUMBER: _ClassVar[int]
    rows: int
    columns: int
    max_turns: int
    def __init__(self, rows: _Optional[int] = ..., columns: _Optional[int] = ..., max_turns: _Optional[int] = ...) -> None: ...

class CreateGameResponse(_message.Message):
    __slots__ = ("game",)
    GAME_FIELD_NUMBER: _ClassVar[int]
    game: GameState
    def __init__(self, game: _Optional[_Union[GameState, _Mapping]] = ...) -> None: ...

class GetGameRequest(_message.Message):
    __slots__ = ("game_id",)
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    game_id: str
    def __init__(self, game_id: _Optional[str] = ...) -> None: ...

class GetGameResponse(_message.Message):
    __slots__ = ("game",)
    GAME_FIELD_NUMBER: _ClassVar[int]
    game: GameState
    def __init__(self, game: _Optional[_Union[GameState, _Mapping]] = ...) -> None: ...

class MakeMoveRequest(_message.Message):
    __slots__ = ("game_id", "expected_turn", "position")
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TURN_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    game_id: str
    expected_turn: int
    position: Coordinate
    def __init__(self, game_id: _Optional[str] = ..., expected_turn: _Optional[int] = ..., position: _Optional[_Union[Coordinate, _Mapping]] = ...) -> None: ...

class MakeMoveResponse(_message.Message):
    __slots__ = ("game", "result")
    GAME_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    game: GameState
    result: MoveResult
    def __init__(self, game: _Optional[_Union[GameState, _Mapping]] = ..., result: _Optional[_Union[MoveResult, _Mapping]] = ...) -> None: ...

class ResetGameRequest(_message.Message):
    __slots__ = ("game_id",)
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    game_id: str
    def __init__(self, game_id: _Optional[str] = ...) -> None: ...

class ResetGameResponse(_message.Message):
    __slots__ = ("game",)
    GAME_FIELD_NUMBER: _ClassVar[int]
    game: GameState
    def __init__(self, game: _Optional[_Union[GameState, _Mapping]] = ...) -> None: ...

class DeleteGameRequest(_message.Message):
    __slots__ = ("game_id",)
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    game_id: str
    def __init__(self, game_id: _Optional[str] = ...) -> None: ...

class DeleteGameResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GameState(_message.Message):
    __slots__ = ("game_id", "board", "scores", "current_player", "turn", "terminal", "termination_reason")
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    BOARD_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PLAYER_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_FIELD_NUMBER: _ClassVar[int]
    TERMINATION_REASON_FIELD_NUMBER: _ClassVar[int]
    game_id: str
    board: Board
    scores: _containers.RepeatedScalarFieldContainer[int]
    current_player: int
    turn: int
    terminal: bool
    termination_reason: TerminationReason
    def __init__(self, game_id: _Optional[str] = ..., board: _Optional[_Union[Board, _Mapping]] = ..., scores: _Optional[_Iterable[int]] = ..., current_player: _Optional[int] = ..., turn: _Optional[int] = ..., terminal: _Optional[bool] = ..., termination_reason: _Optional[_Union[TerminationReason, str]] = ...) -> None: ...

class Board(_message.Message):
    __slots__ = ("rows", "columns", "cells")
    ROWS_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    CELLS_FIELD_NUMBER: _ClassVar[int]
    rows: int
    columns: int
    cells: bytes
    def __init__(self, rows: _Optional[int] = ..., columns: _Optional[int] = ..., cells: _Optional[bytes] = ...) -> None: ...

class MoveResult(_message.Message):
    __slots__ = ("player", "scored_points", "killed_cells", "cordons")
    PLAYER_FIELD_NUMBER: _ClassVar[int]
    SCORED_POINTS_FIELD_NUMBER: _ClassVar[int]
    KILLED_CELLS_FIELD_NUMBER: _ClassVar[int]
    CORDONS_FIELD_NUMBER: _ClassVar[int]
    player: int
    scored_points: int
    killed_cells: _containers.RepeatedCompositeFieldContainer[Coordinate]
    cordons: _containers.RepeatedCompositeFieldContainer[Cordon]
    def __init__(self, player: _Optional[int] = ..., scored_points: _Optional[int] = ..., killed_cells: _Optional[_Iterable[_Union[Coordinate, _Mapping]]] = ..., cordons: _Optional[_Iterable[_Union[Cordon, _Mapping]]] = ...) -> None: ...

class Cordon(_message.Message):
    __slots__ = ("positions",)
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    positions: _containers.RepeatedCompositeFieldContainer[Coordinate]
    def __init__(self, positions: _Optional[_Iterable[_Union[Coordinate, _Mapping]]] = ...) -> None: ...

class Coordinate(_message.Message):
    __slots__ = ("row", "column")
    ROW_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    row: int
    column: int
    def __init__(self, row: _Optional[int] = ..., column: _Optional[int] = ...) -> None: ...
