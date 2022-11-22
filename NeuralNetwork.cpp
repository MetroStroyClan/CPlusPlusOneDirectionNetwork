#include "NeuralNetwork.h"

//static

float nn::Randomizer::rand_float(float min, float max) {

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(min, max);
    return distr(eng);

}

float nn::Activation::Sigm(float v) {

    return (1.0f / (1.0f + (float)(pow(constE, (double)-v))));

}

float nn::Activation::Tanh(float v) {

    return (float)tanh((double)v);

}

//layer abstraction

float *nn::Layer::Propagate(float *inp) {

    return nullptr;

}

float **nn::Layer::GetExternalAxons() {

    return nullptr;

}

float **nn::Layer::GetInnerAxons() {

    return nullptr;

}

void nn::Layer::SetExternalAxons(float **axons) {

}

void nn::Layer::SetInnerAxons(float **axons) {

}

void nn::Layer::MutateAllAxons(float force) {

}

bool nn::Layer::haveInnerAxons() {
    return false;
}

bool nn::Layer::haveHeritageInfo() {
    return false;
}

float **nn::Layer::GetHeritageInfo() {
    return nullptr;
}

void nn::Layer::SetHeritageInfo(float **info) {

}

//basic layer class

nn::BasicLayer::BasicLayer(int count, int nextLayerLength) {

    _count = count;
    _nextLayerLength = nextLayerLength;

    InitAxons();
    FillAxonsRand();

}

nn::BasicLayer::~BasicLayer() {

    DestructAxons();

}

float *nn::BasicLayer::Propagate(float *inp) {

    float* result = new float[_nextLayerLength];

    for(int i = 0; i < _nextLayerLength; i++) {

        result[i] = 0.0f;

        for(int j = 0; j < _count; j++) {

            result[i] += (
                    inp[j] *
                    _axons[j][i]);

        }

        float bv = Activation::Sigm(result[i]);
        result[i] = bv;

    }

    delete[] inp;

    return result;

}

float **nn::BasicLayer::GetExternalAxons() {

    return _axons;

}

float **nn::BasicLayer::GetInnerAxons() {

    return nullptr;

}

void nn::BasicLayer::SetExternalAxons(float **axons) {

    for(int i = 0; i < _count; i++) {

        for(int j = 0; j < _nextLayerLength; j++) {

            _axons[i][j] = axons[i][j];

        }

    }

}

void nn::BasicLayer::SetInnerAxons(float **axons) {

    return;

}

void nn::BasicLayer::MutateAllAxons(float force) {

    for(int i = 0; i < _count; i++) {

        for(int j = 0; j < _nextLayerLength; j++) {

            _axons[i][j] +=
                    nn::Randomizer::rand_float(-1.0f, 1.0f)
                    * force;

        }

    }

}

bool nn::BasicLayer::haveInnerAxons() {

    return false;

}

void nn::BasicLayer::InitAxons() {

    _axons = new float*[_count];

    for(int i = 0; i < _count; i++) {

        _axons[i] = new float[_nextLayerLength];

    }

}

void nn::BasicLayer::DestructAxons() {

    for(int i = 0; i < _count; i++) {

        delete[] _axons[i];

    }

    delete[] _axons;

}

void nn::BasicLayer::FillAxonsRand() {

    for(int i = 0; i < _count; i++) {

        for(int j = 0; j < _nextLayerLength; j++) {

            _axons[i][j] = nn::Randomizer::rand_float(
                    MINIMAL_AXONS_RANDOM_VALUE,
                    MAXIMUM_AXONS_RANDOM_VALUE);

        }

    }

}

bool nn::BasicLayer::haveHeritageInfo() {

    return false;

}

float **nn::BasicLayer::GetHeritageInfo() {

    return nullptr;

}

void nn::BasicLayer::SetHeritageInfo(float **info) {



}

//lstm layer class

nn::LSTM_Layer::LSTM_Layer(int cellState, int inputOnOne,
        int neuronsCount, int nextLayerLength) {

    _cellStateCount = cellState;
    _inputOnOneCount = inputOnOne;
    _neuronsCount = neuronsCount;
    _nextLayerLength = nextLayerLength;

    InitNeurons();
    InitInnerArrays();

    FillAllAxonsRand();

}

nn::LSTM_Layer::~LSTM_Layer() {

    DestructNeurons();
    DestructInnerArrays();

}

float *nn::LSTM_Layer::Propagate(float *inp) {

    _actualInput = new float[_inputOnOneCount * _neuronsCount];

    for(int i = 0; i < _inputOnOneCount * _neuronsCount; i++) {

        _actualInput[i] = inp[i];

    }

    int sumIter = 0;
    float* res = new float[_nextLayerLength];

    for(int i = 0; i < _nextLayerLength; i++) {

        res[i] = 0.0f;

        for(int j = 0; j < _neuronsCount; j++) {

            float* ans = GetAnswerOn(j);

            for(int l = 0; l < _cellStateCount; l++) {

                res[i] += ans[l] *
                        _externalAxons[(j * _cellStateCount) + l][i];

                sumIter++;

            }

            delete[] ans;

        }

        float bv = Activation::Sigm(res[i]);
        res[i] = bv;

    }

    if(_answer == nullptr) {

        _answer = new float[sumIter];

    }

    for(int i = 0; i < sumIter; i++) {

        _answer[i] = res[i];

    }

    delete[] inp;
    delete[] _actualInput;

    return res;

}

float **nn::LSTM_Layer::GetExternalAxons() {

    return _externalAxons;

}

float **nn::LSTM_Layer::GetInnerAxons() {

    return _innerAxons();

}

void nn::LSTM_Layer::SetExternalAxons(float **axons) {

    for(int i = 0; i < _neuronsCount * _cellStateCount; i++) {

        for(int j = 0; j < _nextLayerLength; j++) {

            _externalAxons[i][j] = axons[i][j];

        }

    }

}

void nn::LSTM_Layer::SetInnerAxons(float **axons) {

    for(int i = 0; i < _neuronsCount; i++) {

        _neurons[i].SetAxons(axons[i], _cellStateCount, _inputOnOneCount);

    }

}

void nn::LSTM_Layer::MutateAllAxons(float force) {

    for(int i = 0; i < _neuronsCount; i++) {

        _neurons[i].MutateAxons(force, _cellStateCount, _inputOnOneCount);

    }

    for(int i = 0; i < _neuronsCount * _cellStateCount; i++) {

        for(int j = 0; j < _nextLayerLength; j++) {

            _externalAxons[i][j] +=
                    nn::Randomizer::rand_float(-1.0f, 1.0f) * force;

        }

    }

}

bool nn::LSTM_Layer::haveInnerAxons() {

    return true;

}

void nn::LSTM_Layer::Neuron::InitArrays(int cellStateLen,
                                        int inputCount) {

    fInputWeight = new float[inputCount];
    iInputWeight = new float[inputCount];
    oInputWeight = new float[inputCount];
    cInputWeight = new float[inputCount];

    fOutputWeight = new float[cellStateLen];
    iOutputWeight = new float[cellStateLen];
    oOutputWeight = new float[cellStateLen];
    cOutputWeight = new float[cellStateLen];

}

void nn::LSTM_Layer::Neuron::DestructArrays() {

    delete[] fInputWeight;
    delete[] iInputWeight;
    delete[] oInputWeight;
    delete[] cInputWeight;

    delete[] fOutputWeight;
    delete[] iOutputWeight;
    delete[] oOutputWeight;
    delete[] cOutputWeight;

}

void nn::LSTM_Layer::Neuron::FillAxonsRand(int cellStateLen,
        int inputCount) {

    for(int i = 0; i < inputCount; i++) {

        fInputWeight[i] = Randomizer::rand_float(
                MINIMAL_AXONS_RANDOM_VALUE,
                MAXIMUM_AXONS_RANDOM_VALUE
                );

    }

    for(int i = 0; i < inputCount; i++) {

        iInputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                 MAXIMUM_AXONS_RANDOM_VALUE);

    }

    for(int i = 0; i < inputCount; i++) {

        oInputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                 MAXIMUM_AXONS_RANDOM_VALUE);

    }

    for(int i = 0; i < inputCount; i++) {

        cInputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                 MAXIMUM_AXONS_RANDOM_VALUE);

    }


    for(int i = 0; i < cellStateLen; i++) {

        fOutputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                  MAXIMUM_AXONS_RANDOM_VALUE);

    }

    for(int i = 0; i < cellStateLen; i++) {

        iOutputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                  MAXIMUM_AXONS_RANDOM_VALUE);

    }

    for(int i = 0; i < cellStateLen; i++) {

        oOutputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                  MAXIMUM_AXONS_RANDOM_VALUE);

    }

    for(int i = 0; i < cellStateLen; i++) {

        cOutputWeight[i] = Randomizer::rand_float(MINIMAL_AXONS_RANDOM_VALUE,
                                                  MAXIMUM_AXONS_RANDOM_VALUE);

    }

}

void nn::LSTM_Layer::Neuron::SetAxons(float *rawAxons,
        int cellStateLen, int inputCount) {

    int totalCounter = 0;

    for(int i = 0; i < inputCount; i++) {

        fInputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

    for(int i = 0; i < inputCount; i++) {

        iInputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

    for(int i = 0; i < inputCount; i++) {

        oInputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

    for(int i = 0; i < inputCount; i++) {

        cInputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }


    for(int i = 0; i < cellStateLen; i++) {

        fOutputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

    for(int i = 0; i < cellStateLen; i++) {

        iOutputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

    for(int i = 0; i < cellStateLen; i++) {

        oOutputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

    for(int i = 0; i < cellStateLen; i++) {

        cOutputWeight[i] = rawAxons[totalCounter];
        totalCounter++;

    }

}

void nn::LSTM_Layer::Neuron::MutateAxons(float force,
        int cellStateLen, int inputCount) {

    for(int i = 0; i < inputCount; i++) {

        fInputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

    for(int i = 0; i < inputCount; i++) {

        iInputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

    for(int i = 0; i < inputCount; i++) {

        oInputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

    for(int i = 0; i < inputCount; i++) {

        cInputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }


    for(int i = 0; i < cellStateLen; i++) {

        fOutputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

    for(int i = 0; i < cellStateLen; i++) {

        iOutputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

    for(int i = 0; i < cellStateLen; i++) {

        oOutputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

    for(int i = 0; i < cellStateLen; i++) {

        cOutputWeight[i] +=
                Randomizer::rand_float(-1.0f, 1.0f) * force;

    }

}

float *nn::LSTM_Layer::Neuron::GetRawAxons(int cellStateLen, int inputCount) {

    float* res = new float[(cellStateLen * 4) + (inputCount * 4)];

    int totalCounter = 0;

    for(int i = 0; i < inputCount; i++) {

        res[totalCounter] = fInputWeight[i];
        totalCounter++;

    }

    for(int i = 0; i < inputCount; i++) {

        res[totalCounter] = iInputWeight[i];
        totalCounter++;

    }

    for(int i = 0; i < inputCount; i++) {

        res[totalCounter] = oInputWeight[i];
        totalCounter++;

    }

    for(int i = 0; i < inputCount; i++) {

        res[totalCounter] = cInputWeight[i];
        totalCounter++;

    }


    for(int i = 0; i < cellStateLen; i++) {

        res[totalCounter] = fOutputWeight[i];
        totalCounter++;

    }

    for(int i = 0; i < cellStateLen; i++) {

        res[totalCounter] = fOutputWeight[i];
        totalCounter++;

    }

    for(int i = 0; i < cellStateLen; i++) {

        res[totalCounter] = fOutputWeight[i];
        totalCounter++;

    }

    for(int i = 0; i < cellStateLen; i++) {

        res[totalCounter] = fOutputWeight[i];
        totalCounter++;

    }

    return res;

}

float **nn::LSTM_Layer::_innerAxons() {

    float** res = new float*[_neuronsCount];

    for(int i = 0; i < _neuronsCount; i++) {

        res[i] = _neurons[i].GetRawAxons(
                _cellStateCount, _inputOnOneCount);

    }

    return res;

}

float *nn::LSTM_Layer::GetAnswerOn(int it) {

    float* fVector = GetFVector(it);
    float* iVector = GetIVector(it);
    float* oVector = GetOVector(it);

    float* cVector = GetCVector(it, fVector, iVector);

    if(_prevCellState == nullptr) {

        _prevCellState = new float[_cellStateCount];

    }

    for(int i = 0; i < _cellStateCount; i++) {

        _prevCellState[i] = cVector[i];

    }

    float* res = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        res[i] = cVector[i] * oVector[i];

    }

    delete[] fVector;
    delete[] iVector;
    delete[] oVector;
    delete[] cVector;

    return res;

}

float *nn::LSTM_Layer::GetFVector(int it) {

    float* res = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        for(int j = 0; j < _inputOnOneCount; j++) {

            res[i] += _actualInput[(it * _inputOnOneCount) + j]
                    * _neurons[it].fInputWeight[j];

        }

        for(int j = 0; j < _cellStateCount; j++) {

            if(_answer != nullptr)
                res[i] += _answer[j]
                        * _neurons[it].fOutputWeight[j];

        }

        float bv = Activation::Sigm(res[i]);
        res[i] = bv;

    }

    return res;

}

float *nn::LSTM_Layer::GetIVector(int it) {

    float* res = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        for(int j = 0; j < _inputOnOneCount; j++) {

            res[i] += _actualInput[(it * _inputOnOneCount) + j]
                      * _neurons[it].iInputWeight[j];

        }

        for(int j = 0; j < _cellStateCount; j++) {

            if(_answer != nullptr)
                res[i] += _answer[j]
                        * _neurons[it].iOutputWeight[j];

        }

        float bv = Activation::Sigm(res[i]);
        res[i] = bv;

    }

    return res;

}

float *nn::LSTM_Layer::GetOVector(int it) {

    float* res = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        for(int j = 0; j < _inputOnOneCount; j++) {

            res[i] += _actualInput[(it * _inputOnOneCount) + j]
                      * _neurons[it].oInputWeight[j];

        }

        for(int j = 0; j < _cellStateCount; j++) {

            if(_answer != nullptr)
                res[i] += _answer[j]
                        * _neurons[it].oOutputWeight[j];

        }

        float bv = Activation::Sigm(res[i]);
        res[i] = bv;

    }

    return res;

}

float *nn::LSTM_Layer::GetCVector(int it, float *fVector, float *iVector) {

    float* buffCVector = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        for(int j = 0; j < _inputOnOneCount; j++) {

            buffCVector[i] += _actualInput[(it * _inputOnOneCount) + j]
                      * _neurons[it].cInputWeight[j];

        }

        for(int j = 0; j < _cellStateCount; j++) {

            if(_answer != nullptr)
                buffCVector[i] += _answer[j]
                        * _neurons[it].cOutputWeight[j];

        }

        float bv = Activation::Tanh(buffCVector[i]);
        buffCVector[i] = bv;

    }

    float* buffCIVector = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        buffCIVector[i] =
                buffCVector[i] * iVector[i];

    }

    float* buffFPrevCVector = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        buffFPrevCVector[i] =
                _prevCellState[i] * fVector[i];

    }

    float* res = new float[_cellStateCount];

    for(int i = 0; i < _cellStateCount; i++) {

        res[i] =
                buffFPrevCVector[i] + buffCIVector[i];

    }

    delete[] buffFPrevCVector;
    delete[] buffCVector;
    delete[] buffCIVector;

    return res;

}

void nn::LSTM_Layer::InitNeurons() {

    _neurons = new Neuron[_neuronsCount];

    for(int i = 0; i < _neuronsCount; i++) {

        Neuron neuron;
        neuron.InitArrays(_cellStateCount, _inputOnOneCount);
        _neurons[i] = neuron;

    }

}

void nn::LSTM_Layer::InitInnerArrays() {

    _actualInput = new float[_inputOnOneCount];
    _prevCellState = new float[_cellStateCount];

    _answer = nullptr;

    _externalAxons = new float*[_neuronsCount * _cellStateCount];

    for(int i = 0; i < _neuronsCount * _cellStateCount; i++) {

        _externalAxons[i] = new float[_nextLayerLength];

    }

}

void nn::LSTM_Layer::DestructNeurons() {

    for(int i = 0; i < _neuronsCount; i++) {

        _neurons[i].DestructArrays();

    }

    delete[] _neurons;

}

void nn::LSTM_Layer::DestructInnerArrays() {

    for(int i = 0; i < _neuronsCount * _cellStateCount; i++) {

        delete[] _externalAxons[i];

    }

    delete[] _externalAxons;

    delete[] _answer;
    delete[] _prevCellState;

}

void nn::LSTM_Layer::FillAllAxonsRand() {

    for(int i = 0; i < _neuronsCount; i++) {

        _neurons[i].FillAxonsRand(_cellStateCount, _inputOnOneCount);

    }

    for(int i = 0; i < _neuronsCount * _cellStateCount; i++) {

        for(int j = 0; j < _nextLayerLength; j++) {

            _externalAxons[i][j] = Randomizer::rand_float(
                    MINIMAL_AXONS_RANDOM_VALUE,
                    MAXIMUM_AXONS_RANDOM_VALUE
                    );

        }

    }

}

bool nn::LSTM_Layer::haveHeritageInfo() {

    return true;

}

float **nn::LSTM_Layer::GetHeritageInfo() {

    float** res = new float*[_neuronsCount];

    for(int i = 0; i < _neuronsCount; i++) {

        res[i] = new float[_cellStateCount];

        for(int j = 0; j < _cellStateCount; j++) {

            res[i][j] = _prevCellState[j];

        }

    }

    return res;

}

void nn::LSTM_Layer::SetHeritageInfo(float **info) {

    for(int i = 0; i < _neuronsCount; i++) {

        for(int j = 0; j < _cellStateCount; j++) {

            _prevCellState[j] = info[i][j];

        }

    }

}

//network

NeuralNetwork::NeuralNetwork(std::vector<nn::LayerInfo> layers,
                             int inputCount,
                             int outputCount) {

    _inputCount = inputCount;
    _outputCount = outputCount;

    _layersInfo = layers;

    InitLayers();

}

NeuralNetwork::~NeuralNetwork() {

    DestructLayers();

}

float *NeuralNetwork::Step(float *input) {

    float* copied_input = new float[_inputCount];
    float* result = new float[_outputCount];

    for(int i = 0; i < _inputCount; i++) {

        copied_input[i] = input[i];

    }

    float* localInput = nullptr;

    for(int i = 0; i < _layersInfo.size(); i++) {

        if(i == 0) {

            localInput = copied_input;

        } else if(i + 1 == _layersInfo.size()) {

            float* buffResult = _layers[i]->Propagate(localInput);

            for(int i = 0; i < _outputCount; i++) {

                result[i] = buffResult[i];

            }

            //delete[] localInput;
            break;

        }

        float* buff = _layers[i]->Propagate(localInput);
        localInput = buff;

        buff = nullptr;

    }

    //delete[] copied_input;

    return result;

}

void NeuralNetwork::MutateOn(float force) {

    for(int i = 0; i < _layersInfo.size(); i++) {

        _layers[i]->MutateAllAxons(force);

    }

}

void NeuralNetwork::InitLayers() {

    _layers = new nn::Layer*[_layersInfo.size()];

    for(int i = 0; i < _layersInfo.size(); i++) {

        nn::Layer* layer = nullptr;

        //for last we have exception
        if(i + 1 == _layersInfo.size()) {

            if(_layersInfo[i].type == nn::Basic) {

                layer = new nn::BasicLayer(
                        _layersInfo[i].neuronCount,
                        _outputCount
                        );

            } else {

                layer = new nn::LSTM_Layer(
                        _layersInfo[i].cellStateLength,
                        _layersInfo[i].inputOnOne,
                        _layersInfo[i].neuronCount,
                        _outputCount
                        );

            }

        } else if(_layersInfo[i].type == nn::Basic) {

            layer = new nn::BasicLayer(
                    _layersInfo[i].neuronCount,
                    _layersInfo[i + 1].neuronCount
                    * _layersInfo[i + 1].inputOnOne);

        } else {

            layer = new nn::LSTM_Layer(
                    _layersInfo[i].cellStateLength,
                    _layersInfo[i].inputOnOne,
                    _layersInfo[i].neuronCount,
                    _layersInfo[i + 1].neuronCount *
                    _layersInfo[i + 1].inputOnOne
                    );

        }

        _layers[i] = layer;

    }

}

void NeuralNetwork::DestructLayers() {

    for(int i = 0; i < _layersInfo.size(); i++) {

        delete _layers[i];

    }

    delete[] _layers;

}

float ****NeuralNetwork::GetRawAxons() {

    float**** result = new float***[2];

    result[0] = new float**[_layersInfo.size()];
    for(int i = 0; i < _layersInfo.size(); i++) {

        result[0][i] = _layers[i]->GetExternalAxons();

    }

    result[1] = new float**[_layersInfo.size()];
    for(int i = 0; i < _layersInfo.size(); i++) {

        if(_layers[i]->haveInnerAxons()) {

            result[1][i] = _layers[i]->GetInnerAxons();

        } else {

            result[1][i] = nullptr;

        }

    }

    return result;

}

void NeuralNetwork::SetRawAxons(float ****axons) {

    //external
    for(int i = 0; i < _layersInfo.size(); i++) {

        _layers[i]->SetExternalAxons(axons[0][i]);

    }

    //inner
    for(int i = 0; i < _layersInfo.size(); i++) {

        if(_layers[i]->haveInnerAxons()) {

            _layers[i]->SetInnerAxons(axons[1][i]);

        }

    }

}

float ***NeuralNetwork::GetRawHeritageInfo() {

    int heritageLayersCount = 0;

    for(int i = 0; i < _layersInfo.size(); i++) {

        if(_layersInfo[i].type == nn::LSTM) {

            heritageLayersCount++;

        }

    }

    float*** result = new float**[heritageLayersCount];

    int k = 0;

    for(int i = 0; i < _layersInfo.size(); i++) {

        if(_layersInfo[i].type == nn::LSTM) {

            result[k] = _layers[i]->GetHeritageInfo();
            k++;

        }

    }

    return result;

}

void NeuralNetwork::SetRawHeritageInfo(float ***info) {

    int k = 0;
    for(int i = 0; i < _layersInfo.size(); i++) {

        if(_layersInfo[i].type == nn::LSTM) {

            _layers[i]->SetHeritageInfo(info[k]);
            k++;

        }

    }

}

//cross breeding manager

void h::CrossBreedingManager::Cross(
        NeuralNetwork* parent1,
        NeuralNetwork* parent2,
        NeuralNetwork* child,
        std::vector<nn::LayerInfo> commonInfo,
        int outputSize) {

    float**** bufferAxons = new float***[2];

    float**** firstAxons = parent1->GetRawAxons();
    float**** secondAxons = parent2->GetRawAxons();

    bufferAxons[0] = new float**[commonInfo.size()];
    bufferAxons[1] = new float**[commonInfo.size()];

    for(int i = 0; i < commonInfo.size(); i++) {

        bufferAxons[0][i] = new float*[commonInfo[i].neuronCount
                                       * commonInfo[i].cellStateLength];

        for(int j = 0; j < commonInfo[i].neuronCount
                           * commonInfo[i].cellStateLength; j++) {

            int nextCount = 0;

            if(i + 1 == commonInfo.size()) {

                nextCount = outputSize;

            } else {

                nextCount = commonInfo[i + 1].neuronCount
                            * commonInfo[i + 1].inputOnOne;

            }

            bufferAxons[0][i][j] = new float[nextCount];

            for(int l = 0; l < nextCount; l++) {

                float chance = nn::Randomizer
                ::rand_float(0.0f, 1.0f);

                float first = firstAxons[0][i][j][l];
                float second = secondAxons[0][i][j][l];

                if(chance > 0.5f) {

                    bufferAxons[0][i][j][l] = first;

                } else {

                    bufferAxons[0][i][j][l] = second;

                }

            }

        }

        if(commonInfo[i].type != nn::Basic) {

            bufferAxons[1][i] = new float*[commonInfo[i].neuronCount];

            for(int j = 0; j < commonInfo[i].neuronCount; j++) {

                int nextCount = (
                        commonInfo[i].cellStateLength * 4
                        ) + (commonInfo[i].inputOnOne * 4);

                bufferAxons[1][i][j] = new float[nextCount];

                for(int l = 0; l < nextCount; l++) {

                    float chance = nn::Randomizer
                    ::rand_float(0.0f, 1.0f);

                    float first = firstAxons[1][i][j][l];
                    float second = secondAxons[1][i][j][l];

                    if(chance > 0.5f) {

                        bufferAxons[1][i][j][l] = first;

                    } else {

                        bufferAxons[1][i][j][l] = second;

                    }

                }

            }

        }

    }

    child->SetRawAxons(bufferAxons);

    freeAxons(commonInfo, firstAxons, false);
    freeAxons(commonInfo, secondAxons, false);
    freeAxons(commonInfo, bufferAxons, true); // full: true

}

void h::CrossBreedingManager::SetHeritage(
        NeuralNetwork *parent1,
        NeuralNetwork *parent2,
        NeuralNetwork *child,
        std::vector<nn::LayerInfo> commonInfo) {

    int layersWithHeritage = 0;

    for(int i = 0; i < commonInfo.size(); i++) {

        if(commonInfo[i].type == nn::LSTM) {

            layersWithHeritage++;

        }

    }

    float*** buffHeritage = new float**[layersWithHeritage];

    float*** firstHeritage = parent1->GetRawHeritageInfo();
    float*** secondHeritage = parent2->GetRawHeritageInfo();

    int k = 0;
    for(int i = 0; i < commonInfo.size(); i++) {

        if(commonInfo[i].type == nn::LSTM) {

            buffHeritage[k] = new float*[commonInfo[i].neuronCount];

            for(int j = 0; j < commonInfo[i].neuronCount; j++) {

                buffHeritage[k][j] = new float[commonInfo[i].cellStateLength];

                for(int l = 0; l < commonInfo[i].cellStateLength; l++) {

                    float chance = nn::Randomizer::rand_float(0.0f, 1.0f);

                    if(chance < 0.5f) {

                        buffHeritage[k][j][l] = firstHeritage[k][j][l];

                    } else {

                        buffHeritage[k][j][l] = secondHeritage[k][j][l];

                    }

                }

            }

            k++;

        }

    }

    child->SetRawHeritageInfo(buffHeritage);

    freeHeritage(commonInfo, buffHeritage);
    freeHeritage(commonInfo, firstHeritage);
    freeHeritage(commonInfo, secondHeritage);

}

void h::CrossBreedingManager::freeAxons(
        std::vector<nn::LayerInfo> common,
        float ****array, bool full) {

    if(full) {

        for(int i = 0; i < common.size(); i++) {

            for(int j = 0; j < common[i].cellStateLength *
            common[i].neuronCount; i++) {

                delete [] array[0][i][j];

            }

            if(common[i].type != nn::Basic) {

                int count =
                        (common[i].cellStateLength * 4) +
                                (common[i].inputOnOne * 4);

                for(int j = 0; j < count; j++) {

                    delete [] array[1][i][j];

                }

            }

            delete [] array[0][i];

        }

    }

    for(int k = 0; k < 2; k++) {

        delete [] array[k];

    }

    delete [] array;

}

void h::CrossBreedingManager::freeHeritage(
        std::vector<nn::LayerInfo> common,
        float ***array) {

    for(int i, k = 0; i < common.size(); i++) {

        if(common[i].type == nn::LSTM) {

            for(int j = 0; j < common[i].neuronCount; j++) {

                delete [] array[k][j];

            }

            delete [] array[k];

            k++;

        }

    }

    delete [] array;

}

