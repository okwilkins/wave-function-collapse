from typing import List, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np


class TileType(Enum):
    UNKNOWN = 'U'
    LAND = 'L'
    COAST = 'C'
    SEA = 'S'
    MOUNTAIN = 'M'


@dataclass
class Tile:
    tile_type: TileType
    print_name: str
    allowed: List[TileType]


@dataclass
class MapSpace:
    x: int
    y: int
    tile: Tile
    possibilities: List[Tile]

    @property
    def probabilities(self) -> List[float]:
        # Each probability is equal (for the time being)
        return [1 / len(self.possibilities) for _ in self.possibilities]

    @property
    def shannon_entropy(self) -> float:
        # entropy = - SUM(pi * log(pi)), where pi is the probability of a tile type
        return -np.sum(self.probabilities * np.log(self.probabilities))

    def collapse(self, target_tile: Optional[Tile] = None) -> None:
        if target_tile is None:
            # Randomly select a tile from the possibilities
            target_tile = np.random.choice(self.possibilities)
        else:
            # Check if the target tile is in the possibilities
            if target_tile not in self.possibilities:
                raise ValueError(f'{target_tile} is not in the possibilities')

        self.tile = target_tile
        self.possibilities = [target_tile]


@dataclass
class Map:
    height: int
    width: int
    map_spaces: np.ndarray

    def get_probability_space(self) -> np.array:
        return np.vectorize(lambda x: x.probabilities)(self.map_spaces)

    def get_entropy_space(self) -> np.array:
        return np.vectorize(lambda x: x.shannon_entropy)(self.map_spaces)

    def get_lowest_entropy_space(self) -> MapSpace:
        entropy_space = self.get_entropy_space()
        mask = entropy_space == np.min(entropy_space)
        return np.random.choice(self.map_spaces[mask])

    def propergate_wave(self, start_map_space: MapSpace) -> None:
        check_stack = get_map_space_neighbours(self, start_map_space)


def generate_blank_map(
    height: int,
    width: int,
    blank_tile: Tile,
    blank_possibilities: List[Tile]
) -> Map:
    blank_map_spaces = np.empty((height, width), dtype=object)

    for x in range(width):
        for y in range(height):
            blank_map_spaces[x][y] = MapSpace(
                x=x, y=y,
                tile=blank_tile,
                possibilities=blank_possibilities
            )

    return Map(height=height, width=width, map_spaces=blank_map_spaces)


def get_map_space_neighbours(map_: Map, map_space: MapSpace) -> List[MapSpace]:
    neighbours = []

    if map_space.x - 1 >= 0:
        neighbours.append(map_.map_spaces[map_space.x - 1][map_space.y])
    if map_space.x + 1 < map_.width:
        neighbours.append(map_.map_spaces[map_space.x + 1][map_space.y])
    if map_space.y - 1 >= 0:
        neighbours.append(map_.map_spaces[map_space.x][map_space.y - 1])
    if map_space.y + 1 < map_.height:
        neighbours.append(map_.map_spaces[map_space.x][map_space.y + 1])

    return neighbours


def convert_map_to_display(map_: Map) -> str:
    display = ''
    for y in range(map_.height):
        for x in range(map_.width):
            display += map_.map_spaces[x][y].tile.print_name + ' '
        display += '\n'
    return display


def main():
    blank_tile = Tile(
        tile_type=TileType.UNKNOWN,
        print_name=TileType.UNKNOWN.value,
        allowed=[TileType.LAND, TileType.COAST, TileType.SEA, TileType.MOUNTAIN]
    )
    land_tile = Tile(
        tile_type=TileType.LAND,
        print_name=TileType.LAND.value,
        allowed=[TileType.LAND, TileType.MOUNTAIN]
    )
    coast_tile = Tile(
        tile_type=TileType.COAST,
        print_name=TileType.COAST.value,
        allowed=[TileType.LAND, TileType.SEA]
    )
    sea_tile = Tile(
        tile_type=TileType.SEA,
        print_name=TileType.SEA.value,
        allowed=[TileType.COAST, TileType.SEA]
    )
    mountain_tile = Tile(
        tile_type=TileType.MOUNTAIN,
        print_name=TileType.MOUNTAIN.value,
        allowed=[TileType.LAND, TileType.MOUNTAIN]
    )

    cool_map = generate_blank_map(
        height=10,
        width=10,
        blank_tile=blank_tile,
        blank_possibilities=[land_tile, coast_tile, sea_tile, mountain_tile]
    )
    map_space = cool_map.get_lowest_entropy_space()
    map_space.collapse(target_tile=land_tile)

    print(convert_map_to_display(cool_map))


if __name__ == '__main__':
    main()
