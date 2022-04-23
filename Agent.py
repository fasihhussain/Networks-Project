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
        predeators,
        preys,
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
        self.predeators = predeators
        self.preys = preys

        print(f"Name: {self.name} Predeators: {self.predeators} Preys: {self.preys}")

        self.history = {"B_t": [self.B_t], "B_consumed": []}
        # Flow(prey, predeator)

    def calculate_biomass(self, flow_matrix):
        B_consumed = sum(flow_matrix[:, self.index])
        B_lost = sum(flow_matrix[self.index])

        return np.exp(-self.mu) * self.B_t + ((1 - np.exp(-self.mu)) / self.mu) * (
            self.gammma * B_consumed + self.B_import - B_lost - self.B_export
        )

    def update_biomass(self, flow_matrix):
        self.B_t = self.calculate_biomass(flow_matrix)
        self.history["B_t"].append(self.B_t)
        self.history["B_consumed"].append(list(flow_matrix[:, self.index]))

    def show_history(self, keys=["B_t", "B_consumed"]):
        for key in keys:
            if key == "B_t":
                print(self.history[key])
                plt.plot(self.history[key])
                plt.ylabel("Biomass")
                plt.xlabel("Iteration")
                plt.title(self.name)
                plt.show()
        # plt.plot(self.history["B_t"])
        # plt.ylabel("Biomass")
        # plt.xlabel("Iteration")
        # plt.title(self.name)
        # plt.show()

    def __str__(self):
        return f"Name: {self.name}\nPredeators: {self.predeators}\nPreys: {self.preys}"
