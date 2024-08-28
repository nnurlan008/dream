#include "graph.hpp"
#include <iostream>  // For std::cout
#include <fstream>
#include <iostream>
#include <sstream>


// Default Graph MAX SIZE: 5000 x 5000


Graph::Graph(string graphFilePath) {
	this->graphFilePath = graphFilePath;
	this->hasZeroId = false;
}

string Graph::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

void Graph::readGraph() {
	ifstream infile;
	// printf("reading graphFilePath: %s\n", graphFilePath);
	std::cout << graphFilePath << std::endl;
	infile.open(graphFilePath);

	string line;
	stringstream ss;
	uint edgeCounter = 0;
	uint maxNodeNumber = 0;
	uint minNodeNumber = MAX_DIST;

	Edge newEdge;
	string graphFormat = GetFileExtension(graphFilePath);
	printf("reading starts...\n");
	std::cout << "extension: " << graphFormat << std::endl;
	if(graphFormat == "el" || graphFormat == "wel")
	{
		unsigned int k = 0;
		while (getline(infile, line)) {
			// ignore non graph data
			if (line[0] < '0' || line[0] >'9') {
				// printf("ignored line: %s\n", line);
				std::cout << line << std::endl;

				continue;
			}
			// printf("ignored line: %s\n", line);
			// stringstream ss(line);
			ss.clear();
			ss << line;
			edgeCounter++;
			

			ss >> newEdge.source;
			ss >> newEdge.end;

			if (ss >> newEdge.weight) {
				// load weight 
			}
			else {
				// load default weight
				newEdge.weight = 1;
			}

			// this->graph[start][end] = weight;
			if (newEdge.source == 0){
				this->hasZeroId = true;
			}
			if (newEdge.end == 0){
				this->hasZeroId = true;
			}

			if (maxNodeNumber < newEdge.source) {
				maxNodeNumber = newEdge.source;
			}
			if (maxNodeNumber < newEdge.end) {
				maxNodeNumber = newEdge.end;
			}
			if (minNodeNumber > newEdge.source) {
				minNodeNumber = newEdge.source;
			}
			if (minNodeNumber > newEdge.end) {
				minNodeNumber = newEdge.source;
			}
			

			this->edges.push_back(newEdge);
			// if (k > (1963263821llu/2llu)) break;
			// k++;
		}
	}
	else if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{

	}
	infile.close();

	
	if (this->hasZeroId){
		maxNodeNumber++;
	}
	this->numNodes = maxNodeNumber;
	this->numEdges = edgeCounter;
	this->defaultSource = minNodeNumber;

	std::cout << "Read graph from " << this->graphFilePath << ". This graph contains " << this->numNodes \
		<< " nodes, and " << edgeCounter << " edges" << endl;
}

void Graph::printGraph() {
	// print the graph
	std::cout << "This graph has " << this->numNodes << " nodes and " << this->numEdges << " edges." << endl;
	int size = this->numNodes;

	for (int i = 0; i < this->numEdges; i++){
		Edge edge = edges.at(i);
		std::cout << "Node: " << edge.source << " -> Node: " << edge.end << " Weight: " << edge.weight << endl;
	}
}