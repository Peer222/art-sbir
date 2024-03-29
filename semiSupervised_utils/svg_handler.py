import re
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np


def build_svg(tuple_representation, shape, result_path:str or Path=None) -> str: # returns svg string
    if result_path and type(result_path) == str: result_path = Path(result_path)

    svg_content = f"""<svg width="{shape[0]}" height="{shape[1]}" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n <g display="inline">\n <title>Layer 1</title>\n """
    svg_content += '\n<path d="'
    for tuple in tuple_representation:
        if tuple[2]: svg_content += f"l{tuple[0]},{tuple[1]}" # draw line
        if tuple[3]: svg_content += f"m{tuple[0]},{tuple[1]}" # move position
        if tuple[4]: break
    svg_content += f"""" id="path" stroke-width="2" stroke="#000" fill="none"/>\n"""
    svg_content += """</g>\n </svg>\n"""

    if result_path:
        with open(result_path, 'w') as svg:
            svg.write(svg_content)

    return svg_content

# parses svg from sketchy to semi-supervised fg-sbir format with tuple of 5
def parse_svg(filename:str or Path, result_path:str or Path=None, reduce_factor=1, max_length=100): # returns tuple representaion (result)
    if type(filename) == str: filename = Path(filename)
    if type(result_path) == str: result_path = Path(result_path)

    parsed_paths, shape, erase_flag = create_line_representation(filename)
    result = {'filename': str(filename), 'shape': shape, 'erase_flag': erase_flag, 'max_len':max_length, 'reduce_factor': reduce_factor, 'image':[]}

    x, y = 0, 0

    for i, path in enumerate(parsed_paths):
        for j, token in enumerate(path):
            # if i == len(parsed_paths)-1 and j == len(path)-1: is_end = 1
            # else: is_end = 0

            dx, dy = token.split(',')
            dx, dy = float(dx[1:]), float(dy)

            if 'm' in token: 
                pen_lifted = 1
                pen_touched = 0
                dx, dy = round(dx - x, 5), round(dy - y, 5) # moveto is defined as absolute coordinates
            else: 
                pen_lifted = 0
                pen_touched = 1

            x, y = x + dx, y + dy

            result['image'].append([dx, dy, pen_touched, pen_lifted, 0])

    result['original_length'] = len(result['image'])

    # later added strokes are likely to be more irrelevant 
    if max_length and len(result['image']) > 11 * max_length: result['image'][:10*max_length] # 1000 average is under 900 + margin
    result['image'] = reduce_strokes(result['image'], reduce_factor, max_length)
    if max_length and len(result['image']) > max_length: result['image'] = result['image'][:max_length]
    # result['image'].append([0, 0, 0, 0, 1]) added after model and before loss calculation

    # each pen state sets mode of next stroke instead of current stroke
    for i in range(len(result['image']) - 1):
        result['image'][i][2:] = result['image'][i + 1][2:]

    if result_path:    
        #pickle.dump(result, open(result_path / filename.stem, 'wb'))
        with open(result_path / (filename.stem + '.json'), 'w') as f:
            json.dump(result, f)
    
    return result


def rasterize(tuple_representation, filename) -> None:
    svg = build_svg(tuple_representation)
    # TODO rasterize and save

def load_tuple_representation(filename:str or Path): #
    if type(filename) == str: filename = Path(filename)

    if filename.suffix == '.json': 
        with open(filename, 'r') as f:
            return json.load(f)
    else: 
        with open(filename, 'rb') as f:
            return pickle.load(f)


# vectors are converted from nested list to torch.Tensor
def reshape_vectorSketch(vectorized_sketch, img_width=256, img_height=256):
    vector_sketch = torch.Tensor(vectorized_sketch['image'])
    vector_sketch[:, 0] = vector_sketch[:, 0] / vectorized_sketch['shape'][0] * img_width
    vector_sketch[:, 1] = vector_sketch[:, 1] / vectorized_sketch['shape'][1] * img_height
    vectorized_sketch['original_shape'] = vectorized_sketch['shape']
    vectorized_sketch['shape'] = (img_width, img_height)
    vectorized_sketch['image'] = vector_sketch#[1:] # sketch starts at origin
    return vectorized_sketch
    


# ------ helper functions --------

# if max_length specified strokes are reduced until number of strokes is lower than max_length by factor
def reduce_strokes(sketch, factor, max_length=0):
    if len(sketch) <= max_length: return sketch
    reduced_sketch = []
    i = 0
    while i < len(sketch):
        i_pred = i

        dx, dy = sketch[i][0], sketch[i][1]
        while i + 1 < len(sketch) and sketch[i][2] and sketch[i + 1][2] and i - i_pred < factor:
            i += 1
            dx, dy = dx + sketch[i][0], dy + sketch[i][1]
        reduced_sketch.append([round(dx, 5), round(dy, 5)] + sketch[i_pred][2:5])
        i += 1

    if max_length and factor > 1 and len(reduced_sketch) < len(sketch): reduced_sketch =reduce_strokes(reduced_sketch, factor, max_length)
    return reduced_sketch

def create_line_representation(filename:str or Path='test-bezier.svg') -> List[List[str]]:
    paths, shape, erase = get_paths_from_svg(filename)
    tokenized_paths = [ tokenize_path(path) for path in paths]
    parsed_paths = []
    for path in tokenized_paths:
        parsed_path = [convert_token_to_line(token) for token in path]
        parsed_paths.append(parsed_path)

    return parsed_paths, shape, erase

def get_paths_from_svg(filename:str or Path='test-bezier.svg') -> List[str]:
    with open(filename, 'r') as f:
        lines = [str(line) for line in f.readlines()]
        svg = ""
        for line in lines:
            svg += line

        # gets all paths except for paths with color white (erase): downside erased strokes are included
        paths = re.findall('<path.*?\sd="([^"]+)"[^#]*#000[^/]*/>', svg, re.DOTALL)

        erase_paths = re.findall('<path.*?\sd="([^"]+)"[^#]*#fff[^/]*/>', svg, re.DOTALL)

        shape = get_shape(svg)

        return paths, shape, len(erase_paths)

def get_shape(svg):
    width, height = re.findall('<svg\swidth="(\d+)"\sheight="(\d+)"', svg)[0]
    return int(width), int(height)

def tokenize_path(path:str) -> List[str]:
    splitted_path = path.split('c')

    tokenized_path = []

    for token in splitted_path:
        splitted = token.split('l')
        tokenized_path.extend(splitted)
    return tokenized_path

def convert_token_to_line(token:str) -> str:
    if 'm' in token: return token
    return 'l' + convert_bezier_to_line(token) # lines stay unchanged

def convert_bezier_to_line(substroke:str) -> str:
    return substroke.split(' ')[-1]

# unused
def show_parsed_svg(line_representation:List[List[str]], new_filename:str or Path='test-parsed.svg') -> None:
    svg_content = """<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n <g display="inline">\n <title>Layer 1</title>\n """
    n = 1
    for path in line_representation:
        svg_content += '\n<path d="'
        for token in path:
            svg_content += token
        svg_content += f"""" id="svg_{n}" stroke-width="2" stroke="#000" fill="none"/>\n"""
        n += 1
    svg_content += """</g>\n </svg>\n"""
    with open(new_filename, 'w') as svg:
        svg.write(svg_content)

if __name__ == '__main__':
    tuple_rep = load_tuple_representation('data/sketchy/sketch_vectors_train_100_2/airplane/n02691156_14875-8.json')#parse_svg('../data/sketchy/sketches_svg/zebra/n02391049_9960-5.svg', '.', 5)
    tuple_rep = reshape_vectorSketch(tuple_rep)
    build_svg(tuple_rep['image'], tuple_rep['shape'], '../test_sketches/test-3.svg')
    #line_representation, erased = create_line_representation()

    #show_parsed_svg(line_representation)
