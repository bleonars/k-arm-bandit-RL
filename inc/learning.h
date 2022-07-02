#ifndef __LEARNING_H__
#define __LEARNING_H__

#include <vector>
#include <algorithm>
#include <random>

#include "bandit.h"

class Learning {
public:
    Learning(const std::vector<Bandit> &bandits, const size_t steps = 1000, const bool greedy = true, const float epsilon = 0.1f, const bool optimistic = false, const float optimistic_init = 5.f) : m_bandits{bandits}, m_value_estimate{}, m_num_selected{}, m_steps{steps}, m_greedy{greedy}, m_epsilon{epsilon}, m_random_device{}, m_random_engine{m_random_device()}, m_distribution_greedy{0.f, 1.f}, m_distribution_action{0, m_bandits.size() - 1}, m_avg_reward{} {
        initialize(optimistic, optimistic_init);
    }

    void initialize(const bool optimistic, const float optimistic_init) {
        for (size_t i = 0; i < m_bandits.size(); ++i) {
            if (optimistic)
                m_value_estimate.push_back(optimistic_init);
            else
                m_value_estimate.push_back(0.f);

            m_num_selected.push_back(0);
        }
    }

    void iterate(const float alpha) {
        float total_reward = 0.f;

        for (size_t i = 0; i < m_steps; ++i) {
            size_t action;

            if (m_greedy || (m_distribution_greedy(m_random_engine) > m_epsilon))
                action = static_cast<size_t>(std::distance(m_value_estimate.begin(), std::max_element(m_value_estimate.begin(), m_value_estimate.end())));
            else
                action = m_distribution_action(m_random_engine);

            float reward = m_bandits[action].get_reward();

            m_num_selected[action] += 1;
            m_value_estimate[action] = m_value_estimate[action] + (alpha * (reward - m_value_estimate[action])); // / m_num_selected[action];

            total_reward += reward;
        }

        m_avg_reward = total_reward / m_steps;
    }

    float get_avg_reward() const {
        return m_avg_reward;
    }

private:
    std::vector<Bandit> m_bandits;
    std::vector<float>  m_value_estimate;
    std::vector<size_t> m_num_selected;
    size_t              m_steps;
    bool                m_greedy;
    float               m_epsilon;

    std::random_device                     m_random_device;
    std::mt19937                           m_random_engine;
    std::uniform_real_distribution<float>  m_distribution_greedy;
    std::uniform_int_distribution<size_t>  m_distribution_action;

    float m_avg_reward;
};

#endif
