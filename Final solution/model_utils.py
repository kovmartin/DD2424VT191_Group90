from collections import defaultdict, OrderedDict

def get_layer_size_dict(used_model):
    layer_size_dict = defaultdict(list)
    for layer in used_model.layers:
        layer_size_dict[layer.output_shape[1:3]] += [layer]
    layer_size_dict = OrderedDict(layer_size_dict.items())

    return layer_size_dict

def concat_layers_dict(layer_size_dict):
    outputs = [v[-1].output for k, v in layer_size_dict.items()]
    concat_layers = {layer_size: output for layer_size, output in zip(layer_size_dict.keys(), outputs)}

    return concat_layers

def save_log (save_dir, log):
    with open(save_dir + '/' + 'log.txt', 'w') as log_file:
      for instance in log.keys():
        log_file.writelines(instance)
        log_file.writelines('\n')
        for name, value in zip(log[instance].keys(), log[instance].values()):
            log_file.writelines(str(name) + '|' + str(value))
            log_file.writelines('\n')
        
        log_file.writelines('\n')

    return