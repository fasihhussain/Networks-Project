import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Agent:
    def __init__(
        self,
        index,
        name,
        B,
        gamma,
        mu,
        rho,
        sigma,
        B_import,
        B_export,
        B_refuge,
        predators,
        preys,
        color,
    ):
        """Initialize the Agent

        Args:
            name (str): name of the specie
            B (float): initial biomass
            gamma (float): assimilation efficiency (0 < gamma < 1)
            mu (int): other losses
            rho (float): inertia
            sigma (float): satiation
            B_import (float): import biomass
            B_export (float): export biomass
            B_refuge (float): refuge biomass
        """
        self.index = index
        self.name = name
        self.B_t = B
        self.gammma = gamma
        self.mu = mu
        self.rho = rho
        self.sigma = sigma
        self.B_import = B_import
        self.B_export = B_export
        self.B_refuge = B_refuge
        self.predators = predators
        self.preys = preys
        self.color = color

        print(f"Name: {self.name} Predators: {self.predators} Preys: {self.preys}")

        self.history = {"B_t": [self.B_t], "B_consumed": []}
        # Flow(prey, predator)

    def calculate_biomass(self, flow_matrix, sum_func=sum):
        B_consumed = sum_func(flow_matrix[self.preys, self.index])
        B_lost = sum_func(flow_matrix[self.index, self.predators])

        return np.exp(-self.mu) * self.B_t + ((1 - np.exp(-self.mu)) / self.mu) * (
            self.gammma * B_consumed + self.B_import - B_lost - self.B_export
        )

    def update_biomass(self, flow_matrix):
        self.B_t = self.calculate_biomass(flow_matrix)
        self.history["B_t"].append(self.B_t)
        self.history["B_consumed"].append(
            list(100 * flow_matrix[:, self.index] / sum(flow_matrix[:, self.index]))
        )
        # assert (
        #     sum(flow_matrix[:, self.index]) > 0 or self.name == "Phytoplankton"
        # ), f"consumed biomass of {self.name} is equal to 0"

    def show_history(self, simulation, keys):
        for key, axis in keys.items():
            if key == "B_t":
                # Individual plot
                axis[0].plot(self.history[key], color=self.color)
                axis[0].set_title(self.name)

                # Common plot
                axis[1].plot(self.history[key], label=self.name, color=self.color)
            elif key == "B_consumed":
                history = np.array(self.history["B_consumed"])
                current_level = np.zeros(history.shape[0])
                years = range(1, history.shape[0] + 1)
                axis.set_title(self.name)
                # for i, agent in enumerate(simulation.agents):
                for i in self.preys:
                    agent = simulation.agents[i]
                    axis.fill_between(
                        years,
                        current_level,
                        current_level + history[:, i],
                        color=agent.color,
                        label=agent.name,
                    )
                    current_level += history[:, i]
        # plt.plot(self.history["B_t"])
        # plt.ylabel("Biomass")
        # plt.xlabel("Iteration")
        # plt.title(self.name)
        # plt.show()

    def __str__(self):
        return f"Name: {self.name}\nPredators: {self.predators}\nPreys: {self.preys}"
