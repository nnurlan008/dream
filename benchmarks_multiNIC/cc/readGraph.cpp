#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "graph.h"



void readfile(Graph &G, Graph_m &G_m, int argc, char **argv, uint *u_edgesOffset, \
              uint *u_edgesSize, uint *u_adjacencyList) {

    unsigned int n;
    unsigned int m;

    //If no arguments then read graph from stdin
    printf("argc: %d file: %s\n", argc, argv[8]);
    

    char ch;
    int k = 0;
    // while ((ch = fgetc(fp)) != EOF){
    //     putchar(ch);
    //     if(k > 100) break;
    //     k++;
    // }

    // reading line by line, max 256 bytes
    const unsigned MAX_LENGTH = 256;
    char buffer[MAX_LENGTH];
    // fgets(buffer, MAX_LENGTH, fp);
    // fgets(buffer, MAX_LENGTH, fp);
    // while (fgets(buffer, MAX_LENGTH, fp)){
    //     printf("%s", buffer);
    //     if(k > 100) break;
    //     k++;
    // }

    // FILE *fp;
    uint num_nodes;
    uint num_edges;
    int read_from_file = 0;
    std::ifstream infile (argv[8], ios::in | ios::binary);
    if(argc == 5){
        // fp = fopen(argv[4], "r");
        
        // std::ifstream infile (argv[4], ios::in | ios::binary);
        if (!infile.is_open()) {
            cerr << "Error opening file: " << argv[8] << endl;
            infile.close();
            // exit(-1);
            read_from_file = 0;
        }
        else{
            
            read_from_file = 1;
        }
        
        
        // if (fp == NULL)
        // {
        //     printf("Error: could not open file %s", argv[4]);
        //     read_from_file = false;
        //     fclose(fp);
        // }
        
    }
    printf("read_from_file: %d\n", read_from_file);
    read_from_file = 1;
    if (read_from_file == 1){
        long long int vertex1, vertex2;
        long long int max = 0;
        long long int lines = 0;
        // fgets(buffer, MAX_LENGTH, fp);
        // fgets(buffer, MAX_LENGTH, fp);
        printf("read_from_file: %d max: %lld\n", read_from_file, max);
        std::cout << "Opening file: " << argv[8] << endl;
        
        infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));

        printf("num_nodes: %llu, num_edges: %llu\n", num_nodes, num_edges);

        n = num_nodes;
        m = num_edges;
        G.numVertices = n;
        G.numEdges = m;

        G.edgesOffset_r = new uint[G.numVertices+1];
        // G.edgesSize_r = new uint[G.numVertices];
        G.adjacencyList_r = new uint64_t[G.numEdges];

        // long long unsigned int fixed_size = 2.800*1024*1024*1024ull;
        // void *fixed_ptr;
        // checkError(cudaMalloc(&fixed_ptr, fixed_size));

        // uint oversubs_ratio = 5;
        // long long unsigned int workload = 8414*1024*1024ull;
        // printf("workload: %llu\n",  workload);
        // // workload = 1024*workload;
        // if(oversubs_ratio > 1){
        //     void *over_ptr;
        //     long long unsigned int os_size = workload - (workload / oversubs_ratio);
        //     printf("workload: %llu\n",  workload);
        //     checkError(cudaMalloc(&over_ptr, os_size)); 
        //     printf("os_size: %u\n",  os_size/1024/1024);
        // }
        // exit(0);

        // checkError(cudaMallocManaged(&u_adjacencyList, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_edgesOffset, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_edgesSize, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_distance, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_parent, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_currentQueue, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_nextQueue, sizeof(rdma_buf<int>)));
        // checkError(cudaMallocManaged(&u_degrees, sizeof(rdma_buf<int>)));

        // checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges * sizeof(uint) ));
        // checkError(cudaMallocManaged(&u_edgesOffset, G.numVertices * sizeof(uint) ));
        // checkError(cudaMallocManaged(&u_edgesSize, G.numVertices * sizeof(uint)) );
        // checkError(cudaMallocManaged(&u_distance, G.numVertices * sizeof(uint) ));
        // checkError(cudaMallocManaged(&u_parent, G.numVertices * sizeof(uint) ));
        // checkError(cudaMallocManaged(&u_currentQueue, G.numVertices * sizeof(uint) ));
        // checkError(cudaMallocManaged(&u_nextQueue, G.numVertices * sizeof(uint) ));
        // checkError(cudaMallocManaged(&u_degrees, G.numVertices * sizeof(uint) ));

        unsigned int total_mem = G.numEdges * sizeof(uint) + 7*G.numVertices*sizeof(uint);
        double t_mem = (double) total_mem/(1024*1024*1024); 
        printf("t_mem: %f\n", t_mem);

		infile.read ((char*)G.edgesOffset_r, sizeof(uint)*(num_nodes+1));
		// infile.read ((char*)G.edgesSize_r, sizeof(uint)*num_nodes);
        
		// G_m.nodePointer[num_nodes] = num_edges;

        

        // for (int i = 0; i < n; i++) {
        //     G.edgesOffset.push_back(G.edgesOffset_r[i]);
        //     G.edgesSize.push_back(G.edgesSize_r[i]);
        // }
        // delete G.edgesOffset_r;
        // delete G.edgesSize_r;

        // G.adjacencyList_r = new uint[num_edges];
        // infile.read ((char*)G.adjacencyList_r, sizeof(uint)*num_edges);

        infile.read ((char*)G.adjacencyList_r, sizeof(uint64_t)*num_edges);


        // for(int i = 0; i < m; i++){
        //     G.adjacencyList.push_back(G.adjacencyList_r[i]);
        // }
        // delete G.adjacencyList_r;


        // printf("max: %lld lines: %lld\n", max, lines);
        // max++;
        
        // std::vector<std::vector<int>> adjecancyLists(max);


        // fseek(fp, 0, SEEK_SET);
        // fgets(buffer, MAX_LENGTH, fp);
        // fgets(buffer, MAX_LENGTH, fp);
        // k=0;
        // while (fscanf(fp, "%lld %lld", &vertex1, &vertex2) == 2) {
        //     adjecancyLists[vertex1].push_back(vertex2);
        //     adjecancyLists[vertex2].push_back(vertex1);
        //     // if(k > 100) break;
        //     //     k++;
        // }
        // printf("max: %lld lines: %lld\n", max, lines);
        // fclose(fp);
        
        infile.close();
        printf("File closed!\n");
        // for (int i = 0; i < 100; i++){
        //     // if(G_m.nodePointer[i] == 2){
        //         std::cout << "G_m.nodePointer: " << G_m.nodePointer[i] << "\n";
        //         std::cout << "G_m.edgeList: " << G_m.edgeList[i].end << "\n";
        //     // }
        // }



        // for (int i = 0; i < n; i++) {
        //     G.edgesOffset.push_back(G.adjacencyList.size());
        //     G.edgesSize.push_back(adjecancyLists[i].size());
        //     std::cout << "G.adjacencyList.size(): " << G.adjacencyList.size() << "\n";
        //     std::cout << "adjecancyLists[i].size(): " << adjecancyLists[i].size() << "\n";
        //     for (auto &edge: adjecancyLists[i]) {
        //         std::cout << "edge: " << edge << "\n";
        //         G.adjacencyList.push_back(edge);
        //         std::cout << "G.adjacencyList.data(): " << G.adjacencyList.data()[i] << "\n";
        //     }
        // }

        
        
    }
    else if(read_from_file == 2){
        FILE *fp;
        stringstream ss;
        // infile.close();
        // fp = fopen(argv[4], "r");
        cout << "2. read_from_file: " << read_from_file <<"\n";
        std::string line1;
		int lineNumber = 0;
		int targetLine = 2; // The line number you want to go to

		// Loop through the lines until the target line is reached
		while (lineNumber < targetLine && std::getline(infile, line1)) {
			++lineNumber;
		}
		// getline( infile, line );
		cout << "line:" << line1 <<"\n";
        ss.str("");
        ss.clear();
        ss << line1;
        ss >> num_nodes; // = atoi();
        ss >> num_edges; // = atoi();
        // ss >> num_edges;
        num_edges = 0;
        num_nodes = 0;
        printf("num_nodes: %llu, num_edges: %llu\n", num_nodes, num_edges);
        int k = 0;
        
        long long int vertex1, vertex2;
        while (std::getline( infile, line1 )) {
            ss.str("");
			ss.clear();
			ss << line1;

            ss >> vertex1;
            ss >> vertex2;
            // printf("vertex1: %d, vertex2: %d \n", vertex1, vertex2);
            // break;
            // adjecancyLists[vertex1].push_back(vertex2);
            if(num_nodes < vertex1) num_nodes = vertex1;
            if(num_nodes < vertex2) num_nodes = vertex2;
            num_edges++;
            // // adjecancyLists[vertex2].push_back(vertex1);
            // if(k == num_edges-1) break; 
            //     k++;
        }
        
        num_edges++;
        lineNumber = 0;
		infile.close();
        printf("num_nodes: %llu, num_edges: %llu\n", num_nodes, num_edges);
        std::ifstream infile (argv[4], ios::in | ios::binary);
        // Loop through the lines until the target line is reached
		while (lineNumber < targetLine && std::getline(infile, line1)) {
			++lineNumber;
		}
        // infile.seekg(1);
        cout << "line1: " << line1 <<"\n";
        std::vector<std::vector<int>> adjecancyLists(num_nodes);
        while (std::getline( infile, line1 )) {
            ss.str("");
			ss.clear();
			ss << line1;
            // cout << "line: " << line1 <<"\n";
            ss >> vertex1;
            ss >> vertex2;
            // printf("vertex1: %d, vertex2: %d \n", vertex1, vertex2);
            // break;
            adjecancyLists[vertex1].push_back(vertex2);
            // if(num_nodes < vertex1) num_nodes = vertex1;
            // if(num_nodes < vertex2) num_nodes = vertex2;
            // adjecancyLists[vertex2].push_back(vertex1);
            // if(k == num_edges-1) break; 
            //     k++;
        }
        infile.close();

        printf("num_nodes: %llu, num_edges: %llu\n", num_nodes, num_edges);
        n = num_nodes;

        uint *a1, *a2, *a3, j;
        a1 = new uint[num_nodes];
        a2 = new uint[num_nodes];
        a3 = new uint[num_edges];


        for (int i = 0; i < n; i++) {
            G.edgesOffset.push_back(G.adjacencyList.size());
            G.edgesSize.push_back(adjecancyLists[i].size());
            // a1[i] = a[i];
            // a2[i] = adjecancyLists[i].size();

            // std::cout << "G.adjacencyList.size(): " << G.adjacencyList.size() << "\n";
            // std::cout << "adjecancyLists[i].size(): " << adjecancyLists[i].size() << "\n";
            for (auto &edge: adjecancyLists[i]) {
                // std::cout << "edge: " << edge << "\n";
                
                G.adjacencyList.push_back(edge);
                a3[j] = edge;
                j++;

                // std::cout << "G.adjacencyList.data(): " << G.adjacencyList.data()[i] << "\n";
            }
        }
        G.adjacencyList.clear();

        for (int i = 0; i < n; i++) {
            a1[i] = G.edgesOffset[i];
            a2[i] = G.edgesSize[i];
        }

        G.edgesOffset.clear();
        G.edgesSize.clear();
        
        // for(int i = 0; i < G.adjacencyList.size(); i++){
        //     a3[i] = G.adjacencyList[i];
        // }
        
        // for (auto &edge: adjecancyLists[i]) {
        //         // std::cout << "edge: " << edge << "\n";
                
        //         G.adjacencyList.push_back(edge);
        //         // a3[i] = edge;

        //         // std::cout << "G.adjacencyList.data(): " << G.adjacencyList.data()[i] << "\n";
        // }
        

        for (int i = 0; i < 100; i++){
            
            // if(G.edgesOffset[i] == 2){
                std::cout << "G.edgesOffset: " << a1[i] << "\n";
                std::cout << "  G.edgesSize: " << a2[i] << "\n";
                for (int k = 0; k < a2[i]; k++) {
                    // std::cout << "edge: " << edge << "\n";
                    // G.adjacencyList.push_back(edge);
                    std::cout << "G.adjacencyList: " << a3[a1[i] + k] << "\n";
                } 
                // std::cout << "  G.adjacencyList: " << G.adjacencyList[i] << "\n";
            // }
        }

        string input = string(argv[4]);
        std::ofstream outfile(/*input.substr(0, input.length()-2)+*/"uvm.bcsr", std::ofstream::binary);
		

		outfile.write((char*)&num_nodes, sizeof(unsigned int));
		outfile.write((char*)&num_edges, sizeof(unsigned int));
		outfile.write ((char*)a1, sizeof(unsigned int)*num_nodes);
		outfile.write ((char*)a2, sizeof(unsigned int)*num_nodes);
        outfile.write ((char*)a3, sizeof(unsigned int)*num_edges);

        outfile.close();
        
        exit(0);


    }
    else{ // user-defined or random graph

        bool fromStdin = argc <= 2;
        if (fromStdin) {
            scanf("%d %d", &n, &m);
        } else {
            srand(12345);
            n = atoi(argv[2]);
            m = atoi(argv[3]);
            printf("edges: %s\n", (argv[3]));
            printf("Number of vertices %lld\n", n);
            printf("Number of edges %lld\n\n", m);
        }

        std::vector<std::vector<int> > adjecancyLists(n);
        for (int i = 0; i < m; i++) {
            int u, v;
            if (fromStdin) {
                scanf("%d %d", &u, &v);
                adjecancyLists[u].push_back(v);
            } else {
                u = rand() % n;
                v = rand() % n;
                adjecancyLists[u].push_back(v);
                // if(u == 0 || v == 0)
                // printf("u: %d v: %d\n", u, v);
                adjecancyLists[v].push_back(u);
                // printf("v: %d u: %d\n", v, u);
            }
        }

        for (int i = 0; i < n; i++) {
            G.edgesOffset.push_back(G.adjacencyList.size());
            printf("node: %d, offset: %d\n", i, G.edgesOffset[i]);
            G.edgesSize.push_back(adjecancyLists[i].size());
            printf("    edgeSize[i]: %d\n", G.edgesSize[i]);
            for (auto &edge: adjecancyLists[i]) {
                std::cout << "      edge: " << edge << "\n";
                G.adjacencyList.push_back(edge);
            }
        }

        // for (int i = 0; i < 100; i++){
        //     std::cout << "  G.adjacencyList[i]: " << G.adjacencyList[i] << "\n";
        //     std::cout << "  G.edgesOffset[i]: " << G.edgesOffset[i] << "\n";
        // }

        for (int i = 0; i < n; i++){
            
            if(G.edgesOffset[i] == 1){
                // std::cout << "G.edgesOffset: " << G.edgesOffset[i] << "\n";
                std::cout << "  G.edgesSize: " << G.edgesSize[i] << "\n";
                // std::cout << "G.edgesOffset: " << G.edgesOffset[i] << "\n";
                std::cout << "  G.adjacencyList: " << G.adjacencyList[i] << "\n";
            }
        }

        G.numVertices = n;
        G.numEdges = G.adjacencyList.size();

        printf("Number of vertices %lld\n", G.numVertices);
        printf("Number of edges %lld\n\n", G.numEdges);

    }
}
