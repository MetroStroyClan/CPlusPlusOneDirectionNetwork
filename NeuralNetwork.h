#ifndef GENALGORITHM_G1_NEURALNETWORK_H
#define GENALGORITHM_G1_NEURALNETWORK_H

#include <vector>
#include <math.h>
#include <random>

#define constE 2.71828

#define MINIMAL_AXONS_RANDOM_VALUE -0.1f
#define MAXIMUM_AXONS_RANDOM_VALUE 0.1f

#define DEFAULT_LAYER_INFO_VALUE 1

namespace nn {

    class Randomizer {
    public:

        static float rand_float(float min, float max);

    };

    class Activation {
    public:

        static float Sigm(float v);
        static float Tanh(float v);

    };

    class Layer {
    public:

        Layer() { }
        virtual ~Layer() { }

        virtual float* Propagate(float* inp);

        //RETURN ORIGINAL!
        virtual float** GetExternalAxons();
        virtual float** GetInnerAxons();

        //COPY!
        virtual void SetExternalAxons(float** axons);
        virtual void SetInnerAxons(float** axons);

        virtual void MutateAllAxons(float force);

        virtual bool haveInnerAxons();

        // RETURN CPY | NEED TO FREE
        virtual bool haveHeritageInfo();
        virtual float** GetHeritageInfo();
        virtual void SetHeritageInfo(float** info);

    };

    class BasicLayer : public Layer {
    public:

        BasicLayer(int count,
                int nextLayerLength);
        ~BasicLayer() override;

        float* Propagate(float* inp)
            override ;

        float** GetExternalAxons()
            override ;
        float** GetInnerAxons()
            override ;

        void SetExternalAxons(float** axons)
            override ;
        void SetInnerAxons(float** axons)
            override ;

        void MutateAllAxons(float force)
            override ;

        bool haveInnerAxons()
            override ;

        bool haveHeritageInfo()
            override ;
        float** GetHeritageInfo()
            override ;
        void SetHeritageInfo(float** info)
            override ;

    private:

        float** _axons; // [this] [next]

        int _count;
        int _nextLayerLength;

        void InitAxons();
        void DestructAxons();
        void FillAxonsRand();

    };

    class LSTM_Layer : public Layer {
    public:

        LSTM_Layer(int cellState,
                int inputOnOne,
                int neuronsCount,
                int nextLayerLength);
        ~LSTM_Layer() override;

        float* Propagate(float* inp)
        override ;

        float** GetExternalAxons()
        override ;
        float** GetInnerAxons()
        override ;

        void SetExternalAxons(float** axons)
        override ;
        void SetInnerAxons(float** axons)
        override ;

        void MutateAllAxons(float force)
        override ;

        bool haveInnerAxons()
        override ;

        bool haveHeritageInfo()
        override ;
        float** GetHeritageInfo()
        override ;
        void SetHeritageInfo(float** info)
        override ;

    private:

        struct Neuron {

            float* fInputWeight;
            float* iInputWeight;
            float* oInputWeight;
            float* cInputWeight;

            float* fOutputWeight;
            float* iOutputWeight;
            float* oOutputWeight;
            float* cOutputWeight;

            void InitArrays(int cellStateLen,
                    int inputCount);
            void DestructArrays();

            void FillAxonsRand(int cellStateLen,
                               int inputCount);

            void SetAxons(float* rawAxons,
                    int cellStateLen,
                    int inputCount);
            void MutateAxons(float force,
                    int cellStateLen,
                    int inputCount);
            float* GetRawAxons(int cellStateLen,
                               int inputCount);

        };

        float** _innerAxons ();
        float** _externalAxons; //[_neuronsCount * _cellStateCount] [_nextLayer]

        int _cellStateCount;
        int _inputOnOneCount;
        int _neuronsCount;
        int _nextLayerLength;

        float* _actualInput;
        float* _answer;
        float* _prevCellState;

        Neuron* _neurons;

        //TO DEL!
        float* GetAnswerOn(int it);

        //TO DEL (ALL)!
        float* GetFVector(int it);
        float* GetIVector(int it);
        float* GetOVector(int it);
        float* GetCVector(int it,
                float* fVector, float* iVector);

        void InitNeurons();
        void InitInnerArrays();

        void DestructNeurons();
        void DestructInnerArrays();

        void FillAllAxonsRand();

    };

    enum LayerType {
        Basic, LSTM
    };

    struct LayerInfo {

        LayerType type;

        //DEFAULT VALUE IS 1 !!

        //common
        int neuronCount =
                DEFAULT_LAYER_INFO_VALUE;

        //lstm
        int cellStateLength =
                DEFAULT_LAYER_INFO_VALUE;
        int inputOnOne =
                DEFAULT_LAYER_INFO_VALUE;

    };

}

class NeuralNetwork {
public:

    NeuralNetwork(std::vector<nn::LayerInfo> layers,
            int inputCount, int outputCount);
    ~NeuralNetwork();

    //COPY INPUT / RESULT NEED TO FREE!
    float* Step(float* input);

    void MutateOn(float force);

    // 1: [0 - EXTERNAL / 1 - INNER] 2: [LAYER]
    // 3: [THIS LAYER] 4: [NEXT LAYER]
    // 1-2 lvl need free  3-4 lvl CPY ONLY!
    float****GetRawAxons();
    // cpy only 1-4 lvl  1-2 lvl need free
    void SetRawAxons(float**** axons);

    //1: [LAYER]  2: [NEURON]  3: [DATA]
    //1, 2, 3 lvl do del
    float*** GetRawHeritageInfo();
    //1, 2, 3 lvl cpy, need to del
    void SetRawHeritageInfo(float*** info);

private:

    int _inputCount;
    int _outputCount;

    std::vector<nn::LayerInfo> _layersInfo;
    nn::Layer** _layers;

    void InitLayers();
    void DestructLayers();

};

namespace h {

    class CrossBreedingManager {
    public:

        static void Cross(
                NeuralNetwork* parent1,
                NeuralNetwork* parent2,
                NeuralNetwork* child,
                std::vector<nn::LayerInfo> commonInfo,
                int outputSize);

        static void SetHeritage(
                NeuralNetwork* parent1,
                NeuralNetwork* parent2,
                NeuralNetwork* child,
                std::vector<nn::LayerInfo> commonInfo);

    private:

        static void freeAxons(
                std::vector<nn::LayerInfo> common,
                float**** array,
                bool full);

        static void freeHeritage(
                std::vector<nn::LayerInfo> common,
                float*** array);

    };

}

//TODO fix mem leaks

#endif //GENALGORITHM_G1_NEURALNETWORK_H
