#include "EvictionSetGenerator.cuh"

#define numL2SetsForHashing 1//*modified*8192
//#define firstNumElsTravLoc 1

#define MainBuffSize MB(1024)

int nuwfound = 0;
int numberofways = 0;

void addrHashEvctSetDeterFast(	dataType *mainDevBuff,
				                dataType *mainHstBuff,
                                unsigned int numElsInBuff,
                                dummyType *devRetDummy,
                                timeType *hostTimeBuff,
                                timeType *devTimeBuff,
                                unsigned int *evictIdxHost,
                                unsigned int*evictIdxDev,
                                unsigned int numElsInMainBuffOffset,
                                size_t timeBuffSz,
                                unsigned int numofcollsnaddrskip,
                                unsigned int **indxBuffHost,
                                dataType *targetAddress,
                                unsigned int threshold,
                                unsigned int *perEvSetIdxBuffDev,
                                long unsigned int mainBuffSz);


__global__ void fracGPUEvictSetKern( dataType *mainBuff,
                                    dummyType *devRetDummy,
                                    timeType *mainTimeBuff,
                                    dataType  *targetAddress,
                                    unsigned int *evictIdxBuff,
                                    unsigned int numElsInMainBuffOffset,
                                    unsigned int currNumEls){//,

        dataType *basePtr;
        dataType *otherPtr;
        unsigned int nxtIdx;
        unsigned int nxtIdx_1,prevIdx_1;
        unsigned int tempDummy_1;
        unsigned int start,end;
        float dummyMain_1;
        unsigned int pvtTimeBuff[2];
        __threadfence();
        //printf("Sender Buffer: %p\n",targetAddress);

        tempDummy_1 = 1;
        dummyMain_1 = (float)(1.0*tempDummy_1);
        basePtr = (dataType *)(targetAddress+threadIdx.x);
        __threadfence();

        start = clock();
        nxtIdx =__ldcg((dataType *)basePtr);
        //asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(nxtIdx) : "l"(basePtr));
        tempDummy_1+=nxtIdx;
        end = clock();
        __threadfence();

        dummyMain_1+= (float)(1.0 * (float)tempDummy_1);
        pvtTimeBuff[0] = (unsigned int)(end-start);

        nxtIdx =0;

        #pragma unroll 1
        for(unsigned int k=0;k<1;k++){
                nxtIdx_1 = nxtIdx;
                #pragma unroll 1
                for(unsigned int i=0;i<currNumEls;i++){
                        prevIdx_1 = nxtIdx_1;
                        otherPtr = (dataType *)&mainBuff[nxtIdx_1];
                        nxtIdx_1 = __ldcs((dataType *)otherPtr);
                        tempDummy_1+=nxtIdx_1;
                        __threadfence();
                }
        }

	dummyMain_1+= (float)(1.0 * (float)tempDummy_1);
	__threadfence();

	start = clock();
	nxtIdx =__ldcg((dataType *)basePtr);
	tempDummy_1+=nxtIdx;
	end = clock();
	__threadfence();

	pvtTimeBuff[1] = (unsigned int)(end-start);
	dummyMain_1+=(float)(1.0 * (float)tempDummy_1);

	mainTimeBuff[threadIdx.x] = pvtTimeBuff[0];
	mainTimeBuff[warpSz+threadIdx.x] = pvtTimeBuff[1];
	evictIdxBuff[threadIdx.x] = prevIdx_1;//nxtIdx_1-numElsInMainBuffOffset;
	*devRetDummy=dummyMain_1;
}

__global__ void pointerChaseReformKern(dataType *mainDevBuff,unsigned int *evictIdxBuff,unsigned int numElsInMainOffset){

        unsigned int nxtIdx = mainDevBuff[evictIdxBuff[threadIdx.x]];
        mainDevBuff[evictIdxBuff[threadIdx.x]-numElsInMainOffset] = nxtIdx;
        __threadfence();
}

void hstPtrChaseFunc(dataType *devBuff,unsigned int numOffsetEls,unsigned int numElsInOffset){

        unsigned int tempInd = 0;

        for(unsigned int i=0;i<numOffsetEls-1;i++){

		for(unsigned int j=0;j<warpSz;j++)
			devBuff[tempInd+j] = (i+1)*numElsInOffset+j;

                tempInd = (i+1)*numElsInOffset;
        }
}


/// @brief 
/// @param cacheBuffer 
/// @param targetAddress 
/// @param EvictionSetResult 
extern void EvictionSetGenerator(dataType *cacheBuffer,
                          dataType *targetAddress,
                          dataType *EvictionSetResult){
        //dataType *mainDevBuff;
        dataType *mainHstBuff;
        dummyType *devRetDummy;
        timeType *hostTimeBuff,*devTimeBuff;

        unsigned int *evictIdxHost,*evictIdxDev;
        unsigned int **indxBuffHost;//,*indxBuffDev;
        unsigned int *perEvSetIdxBuffDev;
        unsigned int numElsInBuff;
        unsigned int numElsInMainBuffOffset;
        unsigned int numofcollsnaddrskip;
        unsigned int setOffset;
        unsigned int evThreshold;
        size_t timeBuffSz;
        long unsigned int mainBuffSz;


        unsigned int *page2mb;

        float *timeHost;
        float *timeDev;
        float *dummyVar_Host;
        float *dummyVar_Dev;

        cudaDeviceProp deviceProp;
        size_t l2_size;
        //cudaDeviceReset();



        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        l2_size = deviceProp.l2CacheSize;

        if (l2_size < 128) {
                fprintf(stderr, "Invalid value for GPU_L2_CACHE_LINE_SIZE\n");
                exit(-1);
        }

        evThreshold = CacheMissThershold;
        setOffset = 0;

        evictIdxHost = (unsigned int *)malloc(warpSz*sizeof(unsigned int));
        indxBuffHost = (unsigned int **)malloc(numL2SetsForHashing*sizeof(unsigned int *));

        for(unsigned int i=0;i<numL2SetsForHashing;i++)
                indxBuffHost[i] = (unsigned int *)malloc(numHashedAddr*sizeof(unsigned int));//numHashedAddr defined above
        
        timeBuffSz = 2*warpSz*sizeof(timeType);//MB(timeBuffSzDevMB);//numOfPaddedEvSets*warpSz*sizeof(timeType);
        
        mainBuffSz = MainBuffSize;//CacheBufferMultiplyer*l2_size;//MB(buffSzTotBig);//MB(8192);//
        numElsInBuff = (unsigned int)(mainBuffSz/mainBuffOffsetSz);
        numElsInMainBuffOffset = (unsigned int)(mainBuffOffsetSz/sizeof(dataType));
        numofcollsnaddrskip = 200;//1//

        hostTimeBuff = (timeType *)malloc(timeBuffSz);
        mainHstBuff = (dataType *)malloc(mainBuffSz);

        hstPtrChaseFunc(mainHstBuff, numElsInBuff, numElsInMainBuffOffset);

        //gpuErrchk(cudaMalloc((void **)&mainDevBuff,mainBuffSz));
        gpuErrchk(cudaMalloc((void **)&devRetDummy,sizeof(dummyType)));

        gpuErrchk(cudaMalloc((void **)&devTimeBuff,timeBuffSz));
        gpuErrchk(cudaMalloc((void **)&evictIdxDev,warpSz*sizeof(unsigned int)));
        gpuErrchk(cudaMalloc((void **)&perEvSetIdxBuffDev,numHashedAddr*sizeof(unsigned int)));

        //printf("Sender Buffer: %p\n",targetAddress);

        printf("Eviction set fast generation\n");
        addrHashEvctSetDeterFast(cacheBuffer,
                                mainHstBuff,
                                numElsInBuff,
                                devRetDummy,
                                hostTimeBuff,
                                devTimeBuff,
                                evictIdxHost,
                                evictIdxDev,
                                numElsInMainBuffOffset,
                                timeBuffSz,
                                numofcollsnaddrskip,
                                indxBuffHost,
                                targetAddress,
                                evThreshold,
                                perEvSetIdxBuffDev,
                                mainBuffSz);

        //EvictionSetResult = indxBuffHost[0];

         //for(unsigned int i=0;i<numL2SetsForHashing;i++){

		//printf("\nSet #: %d\n==============\n",i);
        for(unsigned int j=0;j<numHashedAddr;j++)
                EvictionSetResult[j] = indxBuffHost[0][j];
	                //printf("%d\n",indxBuffHost[i][j]);
	//} 


        free(evictIdxHost);
	
        for(unsigned int i=0;i<numL2SetsForHashing;i++)//{
		free(indxBuffHost[i]);
		
	free(indxBuffHost);
	free(hostTimeBuff);
	free(mainHstBuff);

	//gpuErrchk(cudaFree(mainDevBuff));
	gpuErrchk(cudaFree(devRetDummy));
	gpuErrchk(cudaFree(devTimeBuff));
	gpuErrchk(cudaFree(evictIdxDev));
	gpuErrchk(cudaFree(perEvSetIdxBuffDev));
}



/*int main(int argc,char *argv[]){
       
        
    dataType *cacheBuffer;
    unsigned int *EvictionSetResult;

    cudaDeviceProp deviceProp;
    size_t l2_size;



    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    l2_size = deviceProp.l2CacheSize;

    if (l2_size < 128) {
        fprintf(stderr, "Invalid value for GPU_L2_CACHE_LINE_SIZE\n");
        return -1;
    }

    EvictionSetResult = (unsigned int *) malloc(numHashedAddr*sizeof(unsigned int));

    gpuErrchk(cudaMalloc((void **) &cacheBuffer, CacheBufferMultiplyer*l2_size));
    
    EvictionSetGenerator(cacheBuffer,&cacheBuffer[0],EvictionSetResult);

    for(unsigned int j=0;j<numHashedAddr;j++)
	printf("%d\n",EvictionSetResult[j]);

    cudaFree(cacheBuffer);
    free(EvictionSetResult);

	
    return 0;
} */


void evctSetDeterSmall(dataType *mainDevBuff,
                         dummyType *devRetDummy,
                         timeType *hostTimeBuff,
                         timeType *devTimeBuff,
                         dataType  *targetAddress,
                         unsigned int *evictIdxHost,
                         unsigned int *evictIdxDev,
                         unsigned int numElsInMainBuffOffset,
                         unsigned int numAddrLim,
                         size_t timeBuffSz,
                         unsigned int numofcollsnaddrskip,
                         unsigned int threshold){

         unsigned int repeatNum=0;
         bool wrongMiss=false;
         unsigned int numAddrStart = (unsigned int)(numAddrLim-numofcollsnaddrskip);

         for(unsigned int addrCtr=numAddrStart;addrCtr<numAddrLim;addrCtr++){

                 fracGPUEvictSetKern<<<1,warpSz>>>(	mainDevBuff,
							devRetDummy,
							devTimeBuff,
							targetAddress,
							evictIdxDev,
							numElsInMainBuffOffset,
							addrCtr);//,virtAddrBuffDev);
                 cudaDeviceSynchronize();

                 gpuErrchk(cudaMemcpy(hostTimeBuff,devTimeBuff,timeBuffSz,cudaMemcpyDeviceToHost));
                 gpuErrchk(cudaMemcpy(evictIdxHost,evictIdxDev,warpSz*sizeof(unsigned int),cudaMemcpyDeviceToHost));

                 float avgTime[2] = {0,0};
                 for(unsigned int pp=0;pp<warpSz;pp++){
                         avgTime[0]+=hostTimeBuff[pp];
                         avgTime[1]+=hostTimeBuff[warpSz+pp];
                 }

                 avgTime[0] = (float)((float)avgTime[0]/(float)warpSz);
                 avgTime[1] = (float)((float)avgTime[1]/(float)warpSz);

		 if(avgTime[1]>threshold){

                         if(repeatNum==5){
                                 repeatNum=0;
                                 wrongMiss=false;
                                 pointerChaseReformKern<<<1,warpSz>>>(mainDevBuff,evictIdxDev,numElsInMainBuffOffset);
                                 cudaDeviceSynchronize();
                                 return;
                         }
                         else{
                                 //printf("sm=>%d\t%f\t%f\n",addrCtr,avgTime[0],avgTime[1]);
                                 if(wrongMiss==false)
                                         wrongMiss=true;

                                 repeatNum++;
                                 addrCtr--;
                         }
                 }
                 else{
			 //printf("sm %d\t%f\t%f\n",addrCtr,avgTime[0],avgTime[1]);
                         if(wrongMiss){
                                 wrongMiss=false;
                                 repeatNum=0;
                         }
                 }

         }

}

__global__ void WaysGPUEvictSetKern(dataType *mainBuff,
				    dummyType *devRetDummy,
				    timeType *mainTimeBuff,
				    dataType  *targetAddress,
                                    unsigned int *evictIdxBuff,
				    unsigned int numElsInMainBuffOffset,
				    unsigned int currNumEls){//,

        dataType *basePtr;
        dataType *otherPtr;
        unsigned int nxtIdx;
        unsigned int nxtIdx_1,prevIdx_1;
        unsigned int tempDummy_1;
        unsigned int start,end;
        float dummyMain_1;
        unsigned int pvtTimeBuff[2];
        __threadfence();

        tempDummy_1 = 0;
        dummyMain_1 = (float)(1.0*tempDummy_1);
        basePtr = (dataType *)(targetAddress + threadIdx.x);//&mainBuff[tempDummy_1+threadIdx.x];
        __threadfence();

        start = clock();
        nxtIdx =__ldcs((dataType *)basePtr);
        tempDummy_1+=nxtIdx;
        end = clock();
        __threadfence();

        dummyMain_1+= (float)(1.0/(float)tempDummy_1);
        pvtTimeBuff[0] = (unsigned int)(end-start);

        #pragma unroll 1
        for(unsigned int k=0;k<1;k++){
                nxtIdx_1 = nxtIdx;
                #pragma unroll 1
                for(unsigned int i=0;i<currNumEls;i++){
                        prevIdx_1 = nxtIdx_1;
                        otherPtr = (dataType *)&mainBuff[evictIdxBuff[i] + threadIdx.x];
                        nxtIdx_1 = __ldcs((dataType *)otherPtr);
                        tempDummy_1+=nxtIdx_1;
                        __threadfence();
                }
        }

	dummyMain_1+= (float)(1.0/(float)tempDummy_1);
	__threadfence();

	start = clock();
	nxtIdx =__ldcs((dataType *)basePtr);
	tempDummy_1+=nxtIdx;
	end = clock();
	__threadfence();

	pvtTimeBuff[1] = (unsigned int)(end-start);
	dummyMain_1+=(float)(1.0/(float)tempDummy_1);

	mainTimeBuff[threadIdx.x] = pvtTimeBuff[0];
	mainTimeBuff[warpSz+threadIdx.x] = pvtTimeBuff[1];
	
	*devRetDummy=dummyMain_1;
}


void addrHashEvctSetDeterFast(	dataType *mainDevBuff,
                                dataType *mainHstBuff,
                                unsigned int numElsInBuff,
                                dummyType *devRetDummy,
                                timeType *hostTimeBuff,
                                timeType *devTimeBuff,
                                unsigned int *evictIdxHost,
                                unsigned int*evictIdxDev,
                                unsigned int numElsInMainBuffOffset,
                                size_t timeBuffSz,
                                unsigned int numofcollsnaddrskip,
                                unsigned int **indxBuffHost,
                                dataType *targetAddress, //target addr in GPU address space
                                unsigned int threshold,
                                unsigned int *perEvSetIdxBuffDev,
                                long unsigned int mainBuffSz){//,

         unsigned int tempNumElsInBuff = numElsInBuff-1;
         unsigned int repeatNum;
         unsigned int targetIdx;
         unsigned int lnInSetCtr;
         unsigned int i;
	     unsigned int offSet;
         bool wrongMiss;
         FILE *fp;

	     offSet = 0;
             //printf("Sender Buffer: %p\n",targetAddress);
         for(unsigned int setCtr=0;setCtr<numL2SetsForHashing;setCtr++){

		 gpuErrchk(cudaMemcpy(mainDevBuff,mainHstBuff,mainBuffSz,cudaMemcpyHostToDevice));
		
                 repeatNum = 0;
                 //targetIdx = 0;//(setOffset+offSet)*numElsInMainBuffOffset;  
                 lnInSetCtr = 0;
                 i = firstNumElsTrav;
                 wrongMiss=false;
                 printf("Started\n");
		 
                 while(1){

                         if(i==tempNumElsInBuff)
                                 break;
                         if(lnInSetCtr==numHashedAddr)
                                 break;

                         fracGPUEvictSetKern<<<1,warpSz>>>(	mainDevBuff,
								devRetDummy,
								devTimeBuff,
								targetAddress,
								evictIdxDev,
								numElsInMainBuffOffset,
								i);//,virtAddrBuffDev);
                         cudaDeviceSynchronize();

                         gpuErrchk(cudaMemcpy(hostTimeBuff,devTimeBuff,timeBuffSz,cudaMemcpyDeviceToHost));
                         gpuErrchk(cudaMemcpy(evictIdxHost,evictIdxDev,warpSz*sizeof(unsigned int),cudaMemcpyDeviceToHost));

			             float avgTime[2] = {0,0};

                         for(unsigned int pp=0;pp<warpSz;pp++){
                                 avgTime[0]+=hostTimeBuff[pp];
                                 avgTime[1]+=hostTimeBuff[warpSz+pp];
                         }

                         avgTime[0] = (float)((float)avgTime[0]/(float)warpSz);
                         avgTime[1] = (float)((float)avgTime[1]/(float)warpSz);
                         //printf("time0: %f\n", avgTime[0]);
                         //printf("time1: %f\n", avgTime[1]);

                         if(avgTime[1]>threshold){

                                 if(repeatNum==5){

                                         repeatNum=0;
                                         wrongMiss=false;

                                         evctSetDeterSmall(mainDevBuff,
                                                           devRetDummy,
                                                           hostTimeBuff,
                                                           devTimeBuff,
                                                           targetAddress,
                                                           evictIdxHost,
                                                           evictIdxDev,
                                                           numElsInMainBuffOffset,
                                                           i,timeBuffSz,
                                                           numofcollsnaddrskip,
                                                           threshold);//,

                                         indxBuffHost[setCtr][lnInSetCtr]=evictIdxHost[0];
                                         lnInSetCtr++;
                                         printf("%d\n",evictIdxHost[0]);
                                 }
                                 else{
//                                         printf("bg==>%d\t%f\t%f\n",i,avgTime[0],avgTime[1]);
                                         if(wrongMiss==false)
                                                 wrongMiss=true;

                                         repeatNum++;
                                         i=i-numofcollsnaddrskip;
                                 }
                         }
                         else{
  //                               printf("bg %d\t%f\t%f\n",i,avgTime[0],avgTime[1]);
                                 if(wrongMiss){
                                         repeatNum=0;
                                         wrongMiss=false;
                                 }
                         }

                         i=i+numofcollsnaddrskip;
                 }

			offSet+=1;
        }

        unsigned int *evictionDev;
        cudaMalloc((void**)&evictionDev,numHashedAddr*sizeof(unsigned int));
        gpuErrchk(cudaMemcpy(evictionDev,indxBuffHost[0],numHashedAddr*sizeof(unsigned int),cudaMemcpyHostToDevice));

        for(unsigned int setCtr=1;setCtr<numHashedAddr+1;setCtr++){

                 //targetIdx = ConsideredBlock*numElsInMainBuffOffset;
                
                
                WaysGPUEvictSetKern<<<1,warpSz>>>(mainDevBuff,
                                                devRetDummy,
                                                devTimeBuff,
                                                targetAddress,
                                                evictionDev,
                                                numElsInMainBuffOffset,
                                                setCtr);//,virtAddrBuffDev);
                cudaDeviceSynchronize();
                
                

                gpuErrchk(cudaMemcpy(hostTimeBuff,devTimeBuff,timeBuffSz,cudaMemcpyDeviceToHost));
                //gpuErrchk(cudaMemcpy(evictIdxHost,evictIdxDev,warpSz*sizeof(unsigned int),cudaMemcpyDeviceToHost));

                float avgTime[2] = {0,0};

                for(unsigned int pp=0;pp<warpSz;pp++){
                        avgTime[0]+=hostTimeBuff[pp];
                        avgTime[1]+=hostTimeBuff[warpSz+pp];
                }

                avgTime[0] = (float)((float)avgTime[0]/(float)warpSz);
                avgTime[1] = (float)((float)avgTime[1]/(float)warpSz);
                //printf("time0: %f\n", avgTime[0]);
                //printf("%f\n", avgTime[1]);
                

                if(avgTime[1]>threshold){
                        if(!nuwfound)
                        {
                                numberofways = setCtr;
                                nuwfound = 1;
                        }
                        //printf("Number of ways: %d\n",setCtr);

                }

                        /* if(repeatNum==5){

                                repeatNum=0;
                                wrongMiss=false;

                                evctSetDeterSmall(mainDevBuff,
                                                devRetDummy,
                                                hostTimeBuff,
                                                devTimeBuff,
                                                targetIdx,
                                                evictIdxHost,
                                                evictIdxDev,
                                                numElsInMainBuffOffset,
                                                i,timeBuffSz,
                                                numofcollsnaddrskip,
                                                threshold);//,

                                indxBuffHost[setCtr][lnInSetCtr]=evictIdxHost[0];
                                lnInSetCtr++;
                        }
                        else{
//                                         printf("bg==>%d\t%f\t%f\n",i,avgTime[0],avgTime[1]);
                                if(wrongMiss==false)
                                        wrongMiss=true;

                                repeatNum++;
                                i=i-numofcollsnaddrskip;
                        } */
                //}

        

        }
        printf("Determined: Number of ways is %d\n", numberofways);
        
}


































/* __global__ void WaysGPUEvictSetKern(dataType *mainBuff,
				    dummyType *devRetDummy,
				    timeType *mainTimeBuff,
				    unsigned int baseIdx,
                                    unsigned int *evictIdxBuff,
				    unsigned int numElsInMainBuffOffset,
				    unsigned int currNumEls){//,

        dataType *basePtr;
        dataType *otherPtr;
        unsigned int nxtIdx;
        unsigned int nxtIdx_1,prevIdx_1;
        unsigned int tempDummy_1;
        unsigned int start,end;
        float dummyMain_1;
        unsigned int pvtTimeBuff[2];
        __threadfence();

        tempDummy_1 = baseIdx;
        dummyMain_1 = (float)(1.0*baseIdx);
        basePtr = (dataType *)&mainBuff[tempDummy_1+threadIdx.x];
        __threadfence();

        start = clock();
        nxtIdx =__ldcs((dataType *)basePtr);
        tempDummy_1+=nxtIdx;
        end = clock();
        __threadfence();

        dummyMain_1+= (float)(1.0/(float)tempDummy_1);
        pvtTimeBuff[0] = (unsigned int)(end-start);

        #pragma unroll 1
        for(unsigned int k=0;k<1;k++){
                nxtIdx_1 = nxtIdx;
                #pragma unroll 1
                for(unsigned int i=0;i<currNumEls;i++){
                        prevIdx_1 = nxtIdx_1;
                        otherPtr = (dataType *)&mainBuff[evictIdxBuff[i]];
                        nxtIdx_1 = __ldcg((dataType *)otherPtr);
                        tempDummy_1+=nxtIdx_1;
                        __threadfence();
                }
        }

	dummyMain_1+= (float)(1.0/(float)tempDummy_1);
	__threadfence();

	start = clock();
	nxtIdx =__ldcs((dataType *)basePtr);
	tempDummy_1+=nxtIdx;
	end = clock();
	__threadfence();

	pvtTimeBuff[1] = (unsigned int)(end-start);
	dummyMain_1+=(float)(1.0/(float)tempDummy_1);

	mainTimeBuff[threadIdx.x] = pvtTimeBuff[0];
	mainTimeBuff[warpSz+threadIdx.x] = pvtTimeBuff[1];
	
	*devRetDummy=dummyMain_1;
}

__global__ void PrintAccTime(dataType *mainBuff,
                            unsigned int off,
                            unsigned int *evictionset,
                            dummyType *dummyVar
                            ){
        dataType *basePtr;
        dataType *otherPtr;
        unsigned int nxtIdx;
        unsigned int nxtIdx_1,prevIdx_1;
        unsigned int tempDummy_1;
        unsigned int baseIdx = 0;
        unsigned int start,end;
        float dummyMain_1;
        unsigned int pvtTimeBuff[2];
        float t=0;
        __threadfence();

        tempDummy_1 = baseIdx;
        dummyMain_1 = (float)(1.0*baseIdx);
        basePtr = (dataType *)&mainBuff[tempDummy_1+threadIdx.x];
        __threadfence();

        start = clock();
        nxtIdx =__ldcs((dataType *)basePtr);
        tempDummy_1+=nxtIdx;
        end = clock();
        __threadfence();
        t = (float) (end - start);
       // printf("t1: %f\n", t);

        start = clock();
        nxtIdx =__ldcs((dataType *)basePtr);
        tempDummy_1+=nxtIdx;
        end = clock();
        __threadfence();
        t = (float) (end - start);
        //printf("t2: %f\n", t);

        dummyMain_1+= (float)(1.0/(float)tempDummy_1);
        pvtTimeBuff[0] = (unsigned int)(end-start);

        #pragma unroll 1
        for(unsigned int k=0;k<1;k++){
                nxtIdx_1 = nxtIdx;
                #pragma unroll 1
                for(unsigned int i=0;i<numHashedAddr;i++){
                        prevIdx_1 = nxtIdx_1;
                        otherPtr = (dataType *)&mainBuff[evictionset[i]];
                        nxtIdx_1 = __ldcg((dataType *)otherPtr);
                        tempDummy_1+=nxtIdx_1;
                        __threadfence();
                }
        }

	dummyMain_1+= (float)(1.0/(float)tempDummy_1);
	__threadfence();

	start = clock();
	nxtIdx =__ldcs((dataType *)basePtr);
	tempDummy_1+=nxtIdx;
	end = clock();
	__threadfence();
        t = (float) (end - start);
        //printf("t3: %f\n", t);

        

	pvtTimeBuff[1] = (unsigned int)(end-start);
	dummyMain_1+=(float)(1.0/(float)tempDummy_1);

	//mainTimeBuff[threadIdx.x] = pvtTimeBuff[0];
	//mainTimeBuff[warpSz+threadIdx.x] = pvtTimeBuff[1];
	
	*dummyVar=dummyMain_1;

} */
