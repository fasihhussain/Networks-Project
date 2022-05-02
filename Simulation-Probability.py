import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Agent import Agent
from config import agent_configurations, foodweb_configuration


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

        print(f"Name: {self.name} Predators: {self.predators} Preys: {self.preys}")

        self.history = {"B_t": [self.B_t], "B_consumed": []}
        # Flow(prey, predator)

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
        return f"Name: {self.name}\nPredators: {self.predators}\nPreys: {self.preys}"


class Simulation:
    def __init__(self, agent_configurations, foodweb):
        self.foodweb = foodweb
        self.agents = [
            Agent(
                i,
                config["name"],
                config["B"],
                config["gamma"],
                config["mu"],
                config["rho"],
                config["sigma"],
                config["B_import"],
                config["B_export"],
                config["B_refuge"],
                self.get_predators(i),
                self.get_preys(i),
            )
            for i, config in enumerate(agent_configurations)
        ]

    def get_predators(self, agent):
        return list(self.foodweb.neighbors(agent))

    def get_preys(self, agent):
        return list(
            map(
                lambda edge: edge[0],
                filter(lambda edge: edge[1] == agent, self.foodweb.edges()),
            )
        )

    def calculate_flows(self):
        flow_matrix = np.zeros((len(self.agents), len(self.agents)))

        for i, agent in enumerate(self.agents):
            # print(agent.name, agent.preys)
            B_consumed_approx = 0.2 * agent.sigma * agent.B_t
            B_available = np.array(
                list(
                    map(
                        lambda prey: self.agents[prey].B_t - self.agents[prey].B_refuge,
                        agent.preys,
                    )
                )
            )
            B_available_total = sum(B_available)
            calculated_flows = (
                B_available
                * min(B_consumed_approx, B_available_total)
                / B_available_total
            )

            print(agent.name, B_consumed_approx, B_available_total)

            for j, flow in zip(agent.preys, calculated_flows):
                flow_matrix[j, i] = flow
        print(flow_matrix)

        return flow_matrix

    def update_state(self):
        flow_matrix = self.calculate_flows()

        # checking constraints on flow
        for i, agent in enumerate(self.agents):
            if len(agent.preys) != 0:
                B_consumed = sum(flow_matrix[:, i])
                B_lost = sum(flow_matrix[i])

                if B_lost > agent.B_t - agent.B_refuge:
                    flow_matrix[i] *= (agent.B_t - agent.B_refuge) / B_lost

                B_next = agent.calculate_biomass(flow_matrix)

                # excessive loss
                # while (
                #     np.exp(-agent.rho) * agent.B_t
                #     >= B_next - agent.B_import + agent.B_export
                # ):
                #     flow_matrix[i] *= 0.95
                #     B_next = agent.calculate_biomass(flow_matrix)

                # excessive consumption
                # while (
                #     B_next - agent.B_import + agent.B_export
                #     >= np.exp(agent.rho) * agent.B_t
                # ):
                #     flow_matrix[:, i] *= 0.95
                #     B_next = agent.calculate_biomass(flow_matrix)

        for i, agent in enumerate(self.agents):
            agent.update_biomass(flow_matrix)

    def run(self, iterations, show_history=False):
        for i in range(iterations):
            print(f"Running iteration {i}")
            self.update_state()

        if show_history:
            for agent in self.agents:
                agent.show_history(["B_t"])


foodweb = nx.DiGraph()
foodweb.add_nodes_from(range(foodweb_configuration["nodes"]))
foodweb.add_edges_from(foodweb_configuration["edges"])

simulation = Simulation(agent_configurations, foodweb)
simulation.run(50, True)
