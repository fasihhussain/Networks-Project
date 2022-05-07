import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from docplex.mp.model import Model
from docplex.mp.relaxer import Relaxer

from Agent import Agent
from config import agent_configurations, foodweb_configuration


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
                config["color"],
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
        # flow_matrix = np.zeros((len(self.agents), len(self.agents)))
        flow_matrix = [
            [0 for j in range(len(self.agents))] for i in range(len(self.agents))
        ]

        model = Model(name="flow")
        constraints = []

        # Variables
        for prey, predator in self.foodweb.edges():
            flow_matrix[prey][predator] = model.continuous_var(
                name=f"({self.agents[prey].name}->{self.agents[predator].name})"
            )
            constraints.append(
                model.add_constraint(
                    flow_matrix[prey][predator] >= 0.001,
                    ctname=f"({self.agents[prey].name}->{self.agents[predator].name}) >= 0",
                )
            )

        flow_matrix = np.array(flow_matrix)

        # Constraints
        for i, agent in enumerate(self.agents):
            B_next = agent.calculate_biomass(flow_matrix, model.sum)
            B_consumed = model.sum(flow_matrix[agent.preys, i])
            B_lost = model.sum(flow_matrix[i, agent.predators])

            # change in biomass constraint by inertia
            # constraints.append(
            #     model.add_constraint(
            #         np.exp(-agent.rho) * agent.B_t
            #         <= B_next - agent.B_import + agent.B_export,
            #         ctname=f"{agent.name} B_(t+1) lower bound",
            #     )
            # )

            # constraints.append(
            #     model.add_constraint(
            #         B_next - agent.B_import + agent.B_export
            #         <= np.exp(agent.rho) * agent.B_t,
            #         ctname=f"{agent.name} B_(t+1) upper bound",
            #     )
            # )

            # resulting biomass should be greater than B_refuge
            constraints.append(
                model.add_constraint(
                    B_next >= agent.B_refuge, ctname=f"{agent.name} B_(t+1) >= B_refuge"
                )
            )

            # consumed biomass should be less than sigma * B_t
            constraints.append(
                model.add_constraint(
                    B_consumed <= agent.sigma * agent.B_t,
                    ctname=f"{agent.name} consumed biomass",
                )
            )

        for i, constraint in enumerate(constraints):
            print(f"{i}\t{constraint}")

        # Cost
        # Maximize the biomass consumed
        # model.maximize(model.sum(flow_matrix.flatten()))

        # Minimize the ratio between B_t and B_(t+1)
        errors1 = [
            (agent.calculate_biomass(flow_matrix, model.sum) - agent.B_t) / agent.B_t
            for agent in self.agents
        ]
        # model.minimize(model.sum_squares(errors))

        # Maximize the ratio between B_t and B_(t+1)
        B_ratio = [
            agent.calculate_biomass(flow_matrix, model.sum) / agent.B_t
            for agent in self.agents
        ]

        # model.maximize(model.sum(B_ratio))

        # errors = []
        # for i, agent in enumerate(self.agents):
        #     prey_biomass = np.array([self.agents[prey].B_t for prey in agent.preys])
        #     total_prey_biomass = sum([self.agents[prey].B_t for prey in agent.preys])

        #     expected_consumed_ratio = prey_biomass / total_prey_biomass

        #     actual_consumed = flow_matrix[agent.preys, i]
        #     actual_consumed_ratio = actual_consumed / model.sum(actual_consumed)

        #     agent_errors = actual_consumed_ratio - expected_consumed_ratio
        #     errors.append(model.sumsq(agent_errors))

        errors = []
        for i, agent in enumerate(self.agents):
            if not len(agent.preys):
                continue
            prey_biomass = np.array([self.agents[prey].B_t for prey in agent.preys])
            total_prey_biomass = sum([self.agents[prey].B_t for prey in agent.preys])

            expected_consumed_ratio = prey_biomass / total_prey_biomass

            actual_consumed = flow_matrix[agent.preys, i]
            total_actual_consumed = model.sum(actual_consumed)
            # actual_consumed_ratio = actual_consumed / model.sum(actual_consumed)

            agent_errors = np.array(
                [
                    a * total_actual_consumed - b
                    for a, b in zip(expected_consumed_ratio, actual_consumed)
                ]
            )

            # agent_errors = (
            #     expected_consumed_ratio * total_actual_consumed - actual_consumed
            # )

            # agent_errors = (
            #     actual_consumed - expected_consumed_ratio
            # ) / total_prey_biomass

            errors.append(model.sumsq(agent_errors))

        # model.minimize(model.sum(errors))
        model.minimize(model.sum(errors) - 0.1 * model.sum(B_ratio))

        # model.maximize(model.sum([model.sum(errors), model.sum(B_ratio)]))

        # specie i
        # 3 preys: sum of biomass is 100 of the preys
        # prey 1: 50
        # prey 2: 30
        # prey 3: 20

        # optimization function minimize

        # Minimize the ratio between B_t and B_(t+1)
        # model.minimize(model.sum_squares(B_ratio))

        relaxer = Relaxer()
        relaxer.relax(model)

        print(f"Number of Relaxations = {relaxer.number_of_relaxations}")
        relaxer.print_information()

        model.solve()
        print(model.solve_details)

        for prey, predator in self.foodweb.edges():
            flow_matrix[prey, predator] = flow_matrix[prey, predator].solution_value

        flow_matrix = flow_matrix.astype("float64")

        return flow_matrix

    def update_state(self):
        flow_matrix = self.calculate_flows()
        for i, agent in enumerate(self.agents):
            agent.update_biomass(flow_matrix)

    def run(self, iterations, show_history=False):
        for i in range(iterations):
            print(f"Running iteration {i}")
            self.update_state()

        if show_history:
            individual_plots = {
                "B_t": plt.subplots(3, 3),
                "B_consumed": plt.subplots(3, 3),
            }
            B_t_common = plt.subplots()

            for i, agent in enumerate(self.agents):
                agent.show_history(
                    self,
                    {
                        "B_t": [
                            individual_plots["B_t"][1][i % 3, i // 3],
                            B_t_common[1],
                        ],
                        "B_consumed": individual_plots["B_consumed"][1][i % 3, i // 3],
                    },
                )

            # For B_t plots
            individual_plots["B_t"][0].suptitle("Biomass Over Time")
            for ax in individual_plots["B_t"][1][-1]:
                ax.set(xlabel="Iteration (years)")
            for ax in individual_plots["B_t"][1][:, 0]:
                ax.set(ylabel="Biomass (t/km^2)")

            B_t_common[1].set(
                title="Biomass Over Time (All Species)",
                xlabel="Iteration (years)",
                ylabel="Biomass (t/km^2)",
            )
            B_t_common[1].legend(loc="upper right")

            # For B_consumed plots
            individual_plots["B_t"][0].suptitle("Composition of Diet")
            for ax in individual_plots["B_consumed"][1][-1]:
                ax.set(xlabel="Iteration (years)")
            for ax in individual_plots["B_consumed"][1][:, 0]:
                ax.set(ylabel="Percentage")

            handles = [
                mpatches.Patch(label=agent.name, color=agent.color)
                for agent in simulation.agents
            ]
            individual_plots["B_consumed"][0].legend(handles=handles, loc="upper right")
            # lines_labels = [
            #     ax.get_legend_handles_labels()
            #     for ax in individual_plots["B_consumed"][1].flat
            # ]
            # lines, labels = [sum(x, []) for x in zip(*lines_labels)]
            # test = ["red"] * len(labels)
            # print(f"LINES: {lines}")
            # print(f"LABELS: {labels}")
            # # handles, labels = individual_plots["B_consumed"][
            # #     0
            # # ].get_legend_handles_labels()
            # individual_plots["B_consumed"][0].legend(test, labels, loc="upper right")

            plt.show()
            # fig, axs = plt.subplots(3, 3)
            # fig_common, ax_common = plt.subplots()
            # # ax_common.set_title("All Species")
            # for i, agent in enumerate(self.agents):
            #     agent.show_history({"B_t": [axs[i % 3, i // 3], ax_common]})

            # for ax in axs[-1]:
            #     ax.set(xlabel="Iteration (years)")
            # for ax in axs[:, 0]:
            #     ax.set(ylabel="Biomass (t/km^2)")

            # ax_common.set(
            #     title="All Species",
            #     xlabel="Iteration (years)",
            #     ylabel="Biomass (t/km^2)",
            # )
            # ax_common.legend(loc="upper right")
            # # ax.set(xlabel="Iteration (years)", ylabel="Biomass (t/km^2)")
            # # for ax in fig.get_axes():
            # #     ax.label_outer()


if __name__ == "__main__":
    foodweb = nx.DiGraph()
    foodweb.add_nodes_from(range(foodweb_configuration["nodes"]))
    foodweb.add_edges_from(foodweb_configuration["edges"])

    simulation = Simulation(agent_configurations, foodweb)
    simulation.run(50, True)
