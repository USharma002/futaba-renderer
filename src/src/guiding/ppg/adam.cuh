#include "types.cuh"
#include "common.cuh"

struct AdamOptimizer{
    struct State {
        int iter = 0;
        float firstMoment = 0;
        float secondMoment = 0;
        float variable = 0;

        float batchAccumulation = 0;
        float bacthGradient = 0;
    } m_state;

    struct Hyperparameters {
        float learningRate;
        int batchSize;
        float epsilon;
        float beta1;
        float beta2;
    } m_hparams;

    AdamOptimizer(float learningRate, int batchSize = 1, float epsilon = 1e-8f, float beta1 = 0.9f, float beta2 = 0.999f)
        : m_hparams{learningRate, batchSize, epsilon, beta1, beta2} {};
    
    AdamOptimizer& operator=(const AdamOptimizer& arg) {
        m_state = arg.m_state;
        m_hparams = arg.m_hparams;

        return *this;
    }

    AdamOptimizer(const AdamOptimizer& arg) {
        *this = arg;
    }

    void append(float gradient, float statisticalWeight) {
        m_state.batchAccumulation += statisticalWeight;
        m_state.bacthGradient += gradient * statisticalWeight;

        if (m_state.batchAccumulation >= m_hparams.batchSize) {
            step(m_state.bacthGradient / m_state.batchAccumulation);
            m_state.batchAccumulation = 0;
            m_state.bacthGradient = 0;
        }
    }

    void step(float gradient) {
        m_state.iter++;

        m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;

        float biasCorrectedFirstMoment = m_state.firstMoment / (1 - std::pow(m_hparams.beta1, m_state.iter));
        float biasCorrectedSecondMoment = m_state.secondMoment / (1 - std::pow(m_hparams.beta2, m_state.iter));

        m_state.variable -= m_hparams.learningRate * biasCorrectedFirstMoment / (std::sqrt(biasCorrectedSecondMoment) + m_hparams.epsilon);
    }

    float varianble() const {
        return m_state.variable;
    }
};