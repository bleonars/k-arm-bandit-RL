#include <iostream>
#include <vector>
#include <random>
#include <array>

#include "bandit.h"
#include "learning.h"

struct LearningSortContainer {
    float  m_total_reward;
    size_t m_learning_type;
};

bool operator<(const LearningSortContainer &lhs, const LearningSortContainer &rhs) {
    return lhs.m_total_reward < rhs.m_total_reward;
}

int main() {
    constexpr size_t                 num_trials = 2000;
    constexpr size_t                 num_bandits = 10;
    std::vector<Bandit>              k_bandits;
    std::array<size_t, 4>            policy_num_wins{};
    std::array<float, 4>             policy_total_reward{};
    std::random_device               random_device;
    std::mt19937                     random_engine(random_device());
    std::normal_distribution<float>  distribution(0.f, 1.f);

    std::cout << "Running 2000 Different Simulations of 10 Bandit Problems of 1000 steps" << std::endl;

    for (size_t i = 0; i < num_trials; ++i) {
        for (size_t k = 0; k < num_bandits; ++k)
            k_bandits.push_back({distribution(random_engine)});

        Learning greedy(k_bandits, 1000, true, 0.f, false, 0.f); 
        Learning optimistic_greedy(k_bandits, 1000, true, 0.f, true, 1.f);
        Learning epsilon_greedy(k_bandits, 1000, false, 0.0625f, false, 0.f); 
        Learning optimistic_epsilon_greedy(k_bandits, 1000, false, 0.0625f, true, 1.f); 

        greedy.iterate(0.1f);
        optimistic_greedy.iterate(0.1f);
        epsilon_greedy.iterate(0.1f);
        optimistic_epsilon_greedy.iterate(0.1f);

        std::vector<LearningSortContainer> l = {
            {greedy.get_avg_reward(), 0},
            {optimistic_greedy.get_avg_reward(), 1},
            {epsilon_greedy.get_avg_reward(), 2},
            {optimistic_epsilon_greedy.get_avg_reward(), 3}
        };

        std::stable_sort(l.begin(), l.end());
        policy_num_wins[l[3].m_learning_type] += 1;

        policy_total_reward[0] += greedy.get_avg_reward();
        policy_total_reward[1] += optimistic_greedy.get_avg_reward();
        policy_total_reward[2] += epsilon_greedy.get_avg_reward();
        policy_total_reward[3] += optimistic_epsilon_greedy.get_avg_reward();
    }

    std::cout << "Realistic Greedy - Total Wins: " << policy_num_wins[0] << ", Average Reward: " << policy_total_reward[0] / num_trials << std::endl;
    std::cout << "Optimistic Greedy - Total Wins: " << policy_num_wins[1] << ", Average Reward: " << policy_total_reward[1] / num_trials << std::endl;
    std::cout << "Realistic Epsilon Greedy - Total Wins: " << policy_num_wins[2] << ", Average Reward: " << policy_total_reward[2] / num_trials << std::endl;
    std::cout << "Optimistic Epsilon Greedy - Total Wins: " << policy_num_wins[3] << ", Average Reward: " << policy_total_reward[3] / num_trials << std::endl;

    return 0;
}
