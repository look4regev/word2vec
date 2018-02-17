from nn_tools.neural_network import NeuralNetwork


INPUT_LAYER_NEURONS_COUNT = 1000
OUTPUT_LAYER_NEURONS_COUNT = 1000
# Active means hidden layers + output layer
ACTIVE_LAYERS_NEURONS_COUNT = [10, OUTPUT_LAYER_NEURONS_COUNT]
ITERATIONS = 10
L2_REGULARIZATION = 0.1


def main():
    neural_network = NeuralNetwork(INPUT_LAYER_NEURONS_COUNT, ACTIVE_LAYERS_NEURONS_COUNT,
                                   L2_REGULARIZATION)
    del neural_network
    print 'This is still a stub- TBD'


if __name__ == '__main__':
    main()
