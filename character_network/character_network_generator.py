import pandas as pd
import networkx as nx
from pyvis.network import Network
 

class CharacterNetworkGenerator:
    def __init__(self):
        pass


    def generate_character_network(self, df):
        windows = 10
        entity_relationships = []

        for row in df['ners']:
            previous_entities_in_window = []

            for sentence in row:
                previous_entities_in_window.append(list(sentence))
                # Keep only the last 'windows' number of sentences
                previous_entities_in_window = previous_entities_in_window[-windows:]

                # Flatten the 2D list into a 1D list
                flattened_window = [entity for sublist in previous_entities_in_window for entity in sublist]

                for entity in sentence:
                    for entity_in_window in flattened_window:
                        if entity != entity_in_window:
                            entity_relationships.append(sorted([entity, entity_in_window]))

        relationship_df = pd.DataFrame({'value': entity_relationships})
        relationship_df['source'] = relationship_df['value'].apply(lambda x: x[0])
        relationship_df['target'] = relationship_df['value'].apply(lambda x: x[1])
        relationship_df = relationship_df.groupby(['source', 'target']).count().reset_index()  # Count number of times each relationship appears
        relationship_df = relationship_df.sort_values('value', ascending=False)  # Show most common relationships first

        return relationship_df
    
    def draw_network_graph(self, relationship_df):
        relationship_df = relationship_df.sort_values('value', ascending=False)  # Show most common relationships first
        relationship_df = relationship_df.head(200)

        G = nx.from_pandas_edgelist(
            relationship_df,
            source="source",
            target="target",
            edge_attr=["value"],
            create_using=nx.Graph()
        )

        net = Network(notebook=True, width="1000px", height="800px", bgcolor="#222222", font_color="white", cdn_resources="remote")

        node_degree = dict(G.degree)
        nx.set_node_attributes(G, node_degree, "size")
        net.from_nx(G)

        html = net.generate_html()
        html = html.replace("'", "\"")

        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        
        return output_html