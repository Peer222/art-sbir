import re
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict


def build_svg(tuple_representation, result_path:str or Path=None) -> str: # returns svg string
    if result_path and type(result_path) == str: result_path = Path(result_path)

    svg_content = """<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n <g display="inline">\n <title>Layer 1</title>\n """
    svg_content += '\n<path d="'
    for tuple in tuple_representation['image']:
        if tuple[2]: svg_content += f"l{tuple[0]},{tuple[1]}" # draw line
        if tuple[3]: svg_content += f"m{tuple[0]},{tuple[1]}" # move position
    svg_content += f"""" id="path" stroke-width="2" stroke="#000" fill="none"/>\n"""
    svg_content += """</g>\n </svg>\n"""

    if result_path:
        with open(result_path, 'w') as svg:
            svg.write(svg_content)

    return svg_content

# parses svg from sketchy to semi-supervised fg-sbir format with tuple of 5
def parse_svg(filename:str or Path='test-bezier.svg', result_path:str or Path=None): # returns tuple representaion (result)
    if type(filename) == str: filename = Path(filename)
    if type(result_path) == str: result_path = Path(result_path)

    parsed_paths, erase_flag = create_line_representation(filename)
    result = {'filename': str(filename), 'erase_flag': erase_flag, 'image':[]} # (0.0, 0.0, 1,0,0) not sure wether needed or not | pen touched initial state due to paper ???

    x, y = 0, 0

    for i, path in enumerate(parsed_paths):
        for j, token in enumerate(path):
            if i == len(parsed_paths)-1 and j == len(path)-1: is_end = 1
            else: is_end = 0

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

            result['image'].append((dx, dy, pen_touched, pen_lifted, is_end))


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



# ------ helper functions --------

def create_line_representation(filename:str or Path='test-bezier.svg') -> List[List[str]]:
    paths, erase = get_paths_from_svg(filename)
    tokenized_paths = [ tokenize_path(path) for path in paths]
    parsed_paths = []
    for path in tokenized_paths:
        parsed_path = [convert_token_to_line(token) for token in path]
        parsed_paths.append(parsed_path)

    return parsed_paths, erase

def get_paths_from_svg(filename:str or Path='test-bezier.svg') -> List[str]:
    with open(filename, 'r') as f:
        lines = [str(line) for line in f.readlines()]
        svg = ""
        for line in lines:
            svg += line

        # gets all paths except for paths with color white (erase): downside erased strokes are included
        paths = re.findall('<path.*?\sd="([^"]+)"[^#]*#000[^/]*/>', svg, re.DOTALL)

        erase_paths = re.findall('<path.*?\sd="([^"]+)"[^#]*#fff[^/]*/>', svg, re.DOTALL)

        return paths, len(erase_paths)

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
    tuple_rep = parse_svg()
    build_svg(tuple_rep, 'test2.svg')
    #line_representation, erased = create_line_representation()

    #show_parsed_svg(line_representation)
