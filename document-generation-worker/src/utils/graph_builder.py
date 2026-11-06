
from typing import List, Union

import networkx as nx
from haystack import Document
from pyvis.network import Network


class GraphBuilder:
    """
    A utility class for constructing and visualizing directed graphs from document/chunk/file relationships.

    This class uses NetworkX for graph data structures and PyVis for rendering 
    interactive visualizations in HTML format. It can be used to represent 
    relationships such as file → document or parent → child dependencies, 
    with visual differentiation of document types.
    """
    def __init__(
        self,
        graph_file: str = "graph.html",
    ):
        """
        Initialize a new GraphBuilder instance.

        Args:
            graph_file (str): Path or filename where the rendered HTML graph will be saved.
                              Defaults to "graph.html".
        """
        self.graph_file = graph_file

        # Graph structure
        self.G = nx.DiGraph()

        self.color_pool: List[str] = [
            "#6B010B", "#4643FF", "#494B48", "#BD4B00", "#620075", "#998200"
        ]
        self.doc_type_color_tracker: dict[str, str] = {}
        self.doc_type_color_tracker["original"] = "#016B61" #Specific case

    def _color_tracking_doc_type(self, doc_type:str) -> str:
        """
        Retrieve or assign a color for a given document type.
        If the document type is new, it is assigned the next available color from the pool.
        """
        if doc_type in self.doc_type_color_tracker.keys():
            return self.doc_type_color_tracker[doc_type]
        else:
            new_color = self.color_pool.pop(0)
            self.doc_type_color_tracker[doc_type] = new_color
            return new_color

    def add_node(self, id: str, label: str, title: str, color: str = "#707070") -> None:
        """
        Add a single node to the graph.

        Args:
            id : Unique identifier for the node.
            label : Label text displayed on the node.
            title : Tooltip text shown when hovering over the node.
            color : Node color in hex format. Defaults to "#707070" (grey).
        """
        self.G.add_node(
            id,
            label=label,
            color=color,
            title=title
        )
    
    def add_document_nodes(self, documents: List[Document]) -> None:
        """
        Add multiple document nodes to the graph.

        Each node represents a Haystack `Document` and is automatically colored
        based on its document type (found in `doc.meta["type"]` if available).
        """
        for doc in documents:
            if "type" in doc.meta:
                doc_type = doc.meta["type"]
            else:
                doc_type = "Uknown"
            
            color = self._color_tracking_doc_type(doc_type)

            self.G.add_node(
                doc.id,
                label=doc_type,
                color=color,
                title=doc.content
            )

    def add_connection(self, pid: str, did: str) -> None:
        """
        Add edges: pid → did, typically used to represent a parent → child or file → document relationship.
        """
        self.G.add_edge(pid, did)  # parent → child
    
    def add_connections(self, pid: Union[str, List[str]], did: Union[str, List[str]]) -> None:
        """
        Add all edges: (for p in pid) and (d in did) : p → d, typically used to represent parents → childs or files → documents relationship.
        """
        if isinstance(pid, str):
            pid = [pid]
        if isinstance(did, str):
            did = [did]
        
        for p in pid:
            for d in did:
                self.G.add_edge(p, d)  # parent → child

    def save_graph(self) -> None:
        """
        Render the graph as an interactive HTML with persistent info panel.
        """
        # Initialize PyVis network
        net = Network(
            notebook=False,  # Set to False for file output
            height="750px",
            width="100%",
            directed=True
        )
        net.from_nx(self.G)

        # Configure hierarchical layout
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "nodeSpacing": 80,
              "treeSpacing": 300,
              "levelSeparation": 150
            }
          },
          "edges": {
            "smooth": false,
            "arrows": {
              "to": {"enabled": true}
            }
          },
          "physics": {
            "enabled": false
          }
        }
        """)

        # Save initial HTML
        net.save_graph(self.graph_file)

        # Inject persistent info panel
        with open(self.graph_file, "r", encoding="utf-8") as f:
            html = f.read()

        addon = """
<style>
#info-box {
    position: fixed;
    right: 20px;
    top: 20px;
    width: 500px;
    max-height: 400px;
    overflow-y: auto;
    background: #1e1e1e;
    color: #f8f8f2;
    border: 1px solid #444;
    border-radius: 10px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
    padding: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4);
    display: none;
    z-index: 9999;
    white-space: pre-wrap;
}
</style>
<div id="info-box"></div>
<script type="text/javascript">
var infoBox = document.getElementById('info-box');
network.on("selectNode", function (params) {
    var nodeId = params.nodes[0];
    var nodeData = network.body.data.nodes.get(nodeId);
    infoBox.innerHTML = nodeData.title || "<i>No details</i>";
    infoBox.style.display = "block";
});
network.on("deselectNode", function (params) {
    infoBox.style.display = "none";
});
</script>
"""

        html = html.replace("</body>", addon + "\n</body>")

        with open(self.graph_file, "w", encoding="utf-8") as f:
            f.write(html)