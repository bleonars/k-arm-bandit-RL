#ifndef __BANDIT_H__
#define __BANDIT_H__

#include <random>
#include <cmath>

static std::random_device m_random_device;

class Bandit {
public:
    Bandit(const float mean, const float variance = 1.f) : m_random_engine{m_random_device()}, m_distribution{mean, std::sqrt(variance)} {

    }

    float get_reward() {
        return m_distribution(m_random_engine);
    }

    float get_mean() {
        return m_distribution.mean();
    }

    float get_variance() {
        return pow(m_distribution.stddev(), 2.f);
    }

private:
    std::mt19937                    m_random_engine;
    std::normal_distribution<float> m_distribution;
};

#endif
