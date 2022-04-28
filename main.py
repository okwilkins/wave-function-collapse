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

    def __hash__(self):
        return hash((self.x, self.y, self.tile.tile_type))

    def get_probabilities(self) -> List[float]:
        # Each probability is equal (for the time being)
        return [1 / len(self.possibilities) for _ in self.possibilities]

    def get_shannon_entropy(self) -> float:
        # entropy = - SUM(pi * log(pi)), where pi is the probability of a tile type
        return -np.sum(self.get_probabilities() * np.log(self.get_probabilities()))

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
        return np.vectorize(lambda x: x.get_shannon_entropy())(self.map_spaces)

    def get_lowest_entropy_space(self) -> MapSpace:
        entropy_space = self.get_entropy_space()
        mask = entropy_space == np.min(entropy_space)
        return np.random.choice(self.map_spaces[mask])



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


# def get_valid_neighbours(map_: Map, map_space: MapSpace) -> List[MapSpace]:
#     neighbours = get_map_space_neighbours(map_, map_space)

#     for neighbour in neighbours:
#         for neighbour_possibility in neighbour.possibilities:
#             if map_space.tile not in neighbour_possibility:

def update_map_space_possibilities(map_space_1: MapSpace, map_space_2: MapSpace) -> None:
    old_possibilities = map_space_2.possibilities
    # TODO: Fix tile and tiletype mixup
    new_possibilities = set(map_space_1.tile.allowed).intersection(set(map_space_2.possibilities))
    # map_space_2.possibilities = new_possibilities


def convert_map_to_display(map_: Map) -> str:
    display = ''
    for y in range(map_.height):
        for x in range(map_.width):
            display += map_.map_spaces[x][y].tile.print_name + ' '
        display += '\n'
    return display


@dataclass
class MapManager:
    map_: Map
    possible_tiles: List[Tile]

    def propergate_wave(self, start_map_space: MapSpace) -> None:
        check_stack = get_map_space_neighbours(self, start_map_space)
        i = 0

        while len(check_stack) > 0:
            current_intersect = np.intersect1d(
                check_stack[i].possibilities,
                self.possible_tiles
            )

            if current_intersect.size == 0:
                check_stack[i].pop()
                i -= 1


            i += 1


def main():
    blank_tile = Tile(
        tile_type=TileType.UNKNOWN,
        print_name=TileType.UNKNOWN.value,
        allowed=[TileType.LAND, TileType.COAST, TileType.SEA, TileType.MOUNTAIN]
    )
    land_tile = Tile(
        tile_type=TileType.LAND,
        print_name=TileType.LAND.value,
        allowed=[TileType.LAND, TileType.MOUNTAIN, TileType.COAST]
    )
    coast_tile = Tile(
        tile_type=TileType.COAST,
        print_name=TileType.COAST.value,
        allowed=[TileType.COAST, TileType.LAND, TileType.SEA]
    )
    sea_tile = Tile(
        tile_type=TileType.SEA,
        print_name=TileType.SEA.value,
        allowed=[TileType.SEA, TileType.COAST, TileType.SEA]
    )
    mountain_tile = Tile(
        tile_type=TileType.MOUNTAIN,
        print_name=TileType.MOUNTAIN.value,
        allowed=[TileType.MOUNTAIN, TileType.LAND]
    )

    cool_map = generate_blank_map(
        height=5,
        width=5,
        blank_tile=blank_tile,
        blank_possibilities=[land_tile, coast_tile, sea_tile, mountain_tile]
    )
    map_space = cool_map.get_lowest_entropy_space()
    map_space.collapse(target_tile=land_tile)

    neighbour_tiles = get_map_space_neighbours(cool_map, map_space)
    update_map_space_possibilities(map_space, neighbour_tiles[0])


if __name__ == '__main__':
    main()
