
import networkx as nx
import matplotlib.pyplot as plt

#PART B: Tree Construction using networkx
edges=[['Weight','Engine'],['Engine','Turbo'],['Turbo','Fuel Eco'],['Fuel Eco','Fast'],['Weight','Discarded'],['Engine','Discarded'],['Turbo','Discarded'],['Fuel Eco','Discarded'],]
G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
plt.figure()
nx.draw(G,pos,edge_color='black',width=1,linewidths=1,node_size=500,node_color='pink',alpha=0.9,labels={node:node for node in G.nodes()})
nx.draw_networkx_edge_labels(G,pos,edge_labels={('Weight','Engine'):'heavy and average',('Weight','Discarded'):'light',('Engine','Discarded'):'small',('Engine','Turbo'):'large',('Turbo','Discarded'):'yes',('Turbo','Fuel Eco'):'no',('Fuel Eco','Fast'):'bad',('Fuel Eco','Discarded'):'average'},font_color='red')
plt.axis('off')
plt.show()




