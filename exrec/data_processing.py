"""Functions to process data and logs"""
import json
import re


def fix_bracket_data(lines):
    with open('new_file', 'a') as writer:
        for line in lines:
            line = re.sub(r'\(', '[', line)
            line = re.sub(r'\)', ']', line)
            line = line + '\n'
            writer.write(line)
    exit()


def process_data(lines):
    data = []
    for k, line in enumerate(lines):
        line = reformat_line_data_file(line)
        x = reformat_dict(line)
        data.append(x)
    return data


def process_log(lines):
    data = []
    for k, line in enumerate(lines):
        line = reformat_line_log_file(line)
        x = reformat_dict_log(line)
        data.append(x)
    return data


def remove_duplicate(lines):
    seen_lines = set()
    output_lines = []
    for line in lines:
        if line not in seen_lines:
            output_lines.append(line)
            seen_lines.add(line)
    return output_lines


def reformat_line_data_file(line):
    line = re.sub(r':', '\":\"', line)
    subs = re.split(r', ', line)
    new_line = '\"' + subs[0] + '\"'
    for sub in subs[1:]:
        if (sub[0].isnumeric() or sub[0] == '-') and sub[-1].isnumeric():
            new_line += ',' + sub
        elif sub[-1].isnumeric() and '[' in sub:
            new_line += ',\"' + sub
        elif (sub[0].isnumeric() or sub[0] == '-') and ']' in sub:
            new_line += ',' + sub + '\"'
        else:
            new_line += ',\"' + sub + '\"'
    new_line = '{' + new_line + '}'
    return new_line


def reformat_line_log_file(line):
    line = line.strip()
    line = re.sub(r':', '\":\"', line)
    subs = re.split(r', ', line)
    new_line = '\"' + subs[0]
    for sub in subs[1:]:
        if (sub[0].isnumeric() or sub[0] == '-') and sub[-1].isnumeric():
            new_line += ',' + sub
        elif sub[-1].isnumeric() and '[' in sub:
            new_line += ',\"' + sub
        elif (sub[0].isnumeric() or sub[0] == '-') and ']' in sub:
            new_line += ',' + sub + '\"'
        else:
            new_line += ',\"' + sub + '\"'
    new_line = '{' + new_line + '}'
    return new_line


def reformat_dict(dict_str):
    data = json.loads(dict_str)
    data['code_params'] = json.loads(data['code_params'])
    data['meas_params'] = json.loads(data['meas_params'])
    data['noise_params'] = json.loads(data['noise_params'])
    data['init_params'] = json.loads(data['init_params'])
    data['optimal_params'] = json.loads(data['optimal_params'])
    data['encoded_infidelity'] = float(data['encoded_infidelity'])
    data['benchmark_infidelity'] = float(data['benchmark_infidelity'])
    data['diff'] = float(data['diff'])
    if 'ratio' in data.keys():
        data['ratio'] = float(data['ratio'])
    return data


def reformat_dict_log(dict_str):
    data = json.loads(dict_str)
    data['params'] = json.loads(data['params'])
    data['encoded_infidelity'] = float(data['encoded_infidelity'])
    data['benchmark_infidelity'] = float(data['benchmark_infidelity'])
    data['diff'] = float(data['diff'])
    data['ratio'] = float(data['ratio'])
    data['time'] = float(data['time'])
    return data
