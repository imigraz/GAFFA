
import csv
import re
import json
from medical_data_augment_tool.utils.io.common import create_directories_for_file_name


def load_dict_csv(file_name, value_type=str, squeeze=True):
    """
    Loads a .csv file as a dict, where the first column indicate the key string
    and the following columns are the corresponding value or list of values.
    :param file_name: The file name to load.
    :param value_type: Each value will be converted to this type.
    :param squeeze: If true, reduce single entry list to a value.
    :return: A dictionary of every entry of the .csv file.
    """
    d = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_id = row[0]
            value = list(map(value_type, row[1:]))
            if squeeze and len(value) == 1:
                value = value[0]
            d[data_id] = value
    return d


def load_dict_dict_csv(file_name, fieldnames=None):
    """
    Loads a .csv file as a dict, where the first column indicate the key string
    and the following columns are the corresponding values which can be accessed by keys
    defined in the header of the .csv file, or the given header keys.
    :param file_name: The file name to load.
    :param fieldnames: If set, use this list of strings as header and keys, otherwise, use first row of .csv file as keys.
    :return: A dictionary of dictionaries of every entry of the .csv file.
    """
    d = {}
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file, fieldnames=fieldnames)
        for row in reader:
            data_id = next(iter(row.values()))
            d[data_id] = row
    return d


def load_dict_idl(file_name, dim, value_type=str):
    """
    Loads a .idl file as a dict. Returns a list of lists, while dim represents the dimension of the inner list.
    :param file_name: The file name to load.
    :param dim: The dimension of the inner list.
    :param value_type: Each value will be converted to this type.
    :return: A dictionary of every entry of the .idl file.
    """
    numeric_const_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    d = {}
    with open(file_name, 'r') as file:
        for line in file.readlines():
            id_match = re.search('"(.*)"', line)
            id = id_match.groups()[0]
            match_string = '\(' + ','.join(['(' + numeric_const_pattern + ')'] * dim) + '\)'
            coords_matches = re.findall(match_string, line)
            values = []
            for coords_match in coords_matches:
                values.append([value_type(coords_match[i]) for i in range(dim)])
            d[id] = values
    return d


def load_list(file_name, value_type=str):
    """
    Loads a .txt file as a list, where every line is a list entry.
    :param file_name: The file name to load.
    :param value_type: Every list entry is converted to this type.
    :return: A list of every line of the .txt file.
    """
    with open(file_name, 'r') as file:
        return [value_type(line.strip('\n')) for line in file.readlines()]


def load_list_csv(file_name, value_type=str):
    """
    Loads a .csv file as a list of lists, where every line is a list entry.
    :param file_name: The file name to load.
    :param value_type: Every list entry is converted to this type.
    :return: A list of lists of every value of every line of the .csv file.
    """
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        return [list(map(value_type, row)) for row in reader]


def save_dict_csv(d, file_name, header=None):
    """
    Saves a dictionary as a .csv file. The key is written as the first column. If the value is a list or a tuple,
    each entry is written as a consecutive column. Otherwise, the value is written as the second column
    :param d: The dictionary to write
    :param file_name: The file name.
    :param header: If given, this list will be written as a header.
    """
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as file:
        writer = csv.writer(file)
        if header is not None:
            writer.writerow(header)
        for key, value in sorted(d.items()):
            if isinstance(value, list):
                writer.writerow([key] + value)
            elif isinstance(value, tuple):
                writer.writerow([key] + list(value))
            else:
                writer.writerow([key, value])


def save_list_csv(l, file_name, header=None, **kwargs):
    """
    Saves a list as a .csv file. If the list entries are a list or a tuple,
    each entry is written as a consecutive column. Otherwise, the value is written as the second column
    :param l: The (possibly nested) list to write
    :param file_name: The file name.
    :param header: If given, this list will be written as a header.
    """
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as file:
        writer = csv.writer(file, **kwargs)
        if header is not None:
            writer.writerow(header)
        for value in l:
            if isinstance(value, list):
                writer.writerow(value)
            elif isinstance(value, tuple):
                writer.writerow(list(value))
            else:
                writer.writerow([value])


def save_string_txt(string, file_name):
    """
    Saves a string as a text file.
    :param string: The string to write.
    :param file_name: The file name.
    """
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as file:
        file.write(string)


def save_list_txt(string_list, file_name):
    """
    Saves string list as a text file. Each list entry is a new line.
    :param string_list: The string list to write.
    :param file_name: The file name.
    """
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as file:
        string_list_with_endl = [string + '\n' for string in string_list]
        file.writelines(string_list_with_endl)


def save_json(obj, file_name, *args, **kwargs):
    """
    Saves an object as a json file.
    :param obj: The object to save.
    :param file_name: The filename.
    :param args: args to pass to json.dump()
    :param kwargs: kwargs to pass to json.dump()
    :return:
    """
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as f:
        json.dump(obj, f, *args, **kwargs)
