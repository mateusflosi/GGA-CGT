/************************************************************************************************************************
 The program excecutes GGA-CGT over a set instances using different configurations                                      *
 given by the user. Each configuration represents an independent execution of the GA.                                   *
                                                                                                                        *
 Reference:                                                                                                             *
        Quiroz-Castellanos, M., Cruz-Reyes, L., Torres-Jimenez, J.,                                                     *
        G�mez, C., Huacuja, H. J. F., & Alvim, A. C. (2015).                                                            *
      A grouping genetic algorithm with controlled gene transmission for                                                *
      the bin packing problem. Computers & Operations Research, 55, 52-64.                                              *
                                                                                                                        *
 Input:                                                                                                                 *
    File "instances.txt" including the name of the BPP instances to be solved;                                          *
    Files including the standard instances to be solve;                                                                 *
    File "configurations.txt" including the parameter values for each experiment;                                       *
                                                                                                                        *
 Output:                                                                                                                *
    A set of files "GGA-CGT_(i).txt" including the experimental results for each                                        *
    configuration i in the input, stored in directory: Solutions_GGA-CGT;                                               *
   If(save_bestSolution = 1) a set of files HGGA_S_(i)_instance.txt including the                                       *
   obtained solution for each instance, for each configuration i, stored in directory: Details_GGA-CGT;                 *
************************************************************************************************************************/

/************************************************************************************************************************
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! *
 * !!!!!!!  NOTE ON MPI_ON: There should be at least one configuration and there should be at least one instance to work *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! *
 ************************************************************************************************************************/

//#include "linked_list.h"
#include <exception>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <sstream>
#include <fstream>
#include <map>
#include <list>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include "includes.h"

// CONSTANTS DEFINING THE SIZE OF THE PROBLEM
#define ATTRIBUTES 5000
#define P_size_MAX 6000
#define Max_items 1000
#define MAX_QUEUE_SIZE 150 // Maximum number of unique schedule steps that the scheduler can execute

#define PRINT_POPULATION_CONTENTS
#undef PRINT_POPULATION_CONTENTS

#define VERBOSE
#undef VERBOSE
#define VERBOSE_SCHEDULE
#undef VERBOSE_SCHEDULE
#define UNPACK
//#undef UNPACK

#define SAME_SEED_ON_ALL_PROCS
#undef SAME_SEED_ON_ALL_PROCS

/****************************************************************************
 * The fundamental difference between a master node and a normal node is:   *
 * That, a normal node always make one connection and communication with    *
 * some other node in the schedule file                                     *
 *                                                                          *
 * However, MASTER_NODE_ENABLED mode allows you to write a specific         *
 * communication routine where a node in a schedule step can make more      *
 * than one communication, or may be able to BCast information to all nodes *
 *                                                                          *
 * if not specified, all processors will adhere to the communication        *
 * schedule given in schedule.txt                                           *
 * ************************************************************************ */
#define MASTER_NODE_ENABLED
#undef MASTER_NODE_ENABLED

/*************************************************************/
/*                 STRUCTS                                   */
/*************************************************************/
struct SOLUTION
{
    linked_list L;
    double Bin_Fullness;
};

struct ScheduleQueue
{
    int *schedule[MAX_QUEUE_SIZE];
    int scheduleSizes[MAX_QUEUE_SIZE];
    int size;
    int start;
    int end;
};

/*************************************************************/
/*                 GLOBALS                                   */
/*************************************************************/
using namespace std;
char file[50];
char nameC[50];
char scheduleFileName[50];

int is_optimal_solution;
int save_bestSolution;
int generation;
int repeated_fitness;
int max_gen;
int life_span;
int P_size;

// Initial seeds for the random number generation
int seed_emptybin;
int seed_permutation;

int *receiveProcs;
int *sendProcs;
int receiveProcCount;
int sendProcCount;

long int i;
long int j;
long int k;
long int l;
long int conf;
long int number_items;
long int bin_capacity;
long int best_solution;
long int n_;
long int L2;
long int bin_i;
long int higher_weight;
long int lighter_weight;

long int ordered_weight[ATTRIBUTES];
long int weight[ATTRIBUTES];
long int permutation[ATTRIBUTES];
long int items_auxiliary[ATTRIBUTES];
long int ordered_population[P_size_MAX];
long int best_individuals[P_size_MAX];
long int random_individuals[P_size_MAX];

double p_m;
double p_c;
double k_ncs;
double k_cs;
double B_size;
double TotalTime;

double total_accumulated_weight;
double weight1[ATTRIBUTES];
double _p_;

clock_t start, end;

SOLUTION global_best_solution[ATTRIBUTES];
SOLUTION population[P_size_MAX][ATTRIBUTES];
SOLUTION children[P_size_MAX][ATTRIBUTES];

FILE *output;
FILE *input_Configurations;
FILE *input_Instances;
FILE *input_Schedule;

// Scheduler Variables
ScheduleQueue Send_Schedules;
ScheduleQueue Receive_Schedules;

// Scheduler Components
void Read_Schedule(char *scheduleFile);
void initQueue(ScheduleQueue *queue, int size);
int getQueueItem(ScheduleQueue *queue);
void queueNext(ScheduleQueue *queue);

int numScheduleSteps;

// GA COMPONENTS
long int Generate_Initial_Population();
long int Generation();
void Gene_Level_Crossover_FFD(long int, long int, long int);
void Adaptive_Mutation_RP(long int, float, int);
void FF_n_(int);                                     // First Fit with � pre-allocated-items (FF-�)
void RP(long int, long int &, long int[], long int); // Rearrangement by Pairs

// BPP Procedures
void FF(long int, SOLUTION[], long int &, long int, int);
void LowerBound();

// Auxiliary functions
void Find_Best_Solution();
void Sort_Ascending_IndividualsFitness();
void Sort_Descending_Weights(long int[], long int);
void Sort_Ascending_BinFullness(long int[], long int);
void Sort_Descending_BinFullness(long int[], long int);
void Sort_Random(long int[], long int, int);
void Copy_Solution(SOLUTION[], SOLUTION[], int);
void Clean_population();
long int Used_Items(long int, long int, long int[]);
void Adjust_Solution(long int);
long int LoadData();
void WriteOutput();
void sendtofile(SOLUTION[]);
void printAllPopulation(SOLUTION *, FILE *);

// Pseudo-random number generator functions
int get_rand_ij(int *, int, int);
int get_rand(int *, int);
float randp(int *);
int trand();

// comparison functions
int compare(double comparedBinFullness);       // returns 0 if different, 1 otherwise
int compare(SOLUTION *comparedPopulationItem); // returns 0 if different, 1 otherwise

#ifdef MPI_ON
                                               // MPI parameters
int xcount = 0;
int count;
int tcount;
int totalCount;
int size, myrank;
int configSignal;   // for broadcasting whether there will be an ongoing config test or not.
int instanceSignal; // for broadcasting whether there will be an ongoing instance test or not.

MPI_Status status;
MPI_Status *statuses;
MPI_Request *recvRequests;

node *temp;
int MaxCommsPerScheduleStep;

int whereToWritePopulation[P_size_MAX];

int isPer;
int numberOfItemsToSend;
int epochLength;
int BorW;
int fullHalt;
int iteration;

void readHPCSchedule(char *fileName)
{
    char buffer[200];
    FILE *HPCScheduleFile;
    char *l, *t;

    if (myrank == 0)
    {
        HPCScheduleFile = fopen(fileName, "r");
        if (!HPCScheduleFile)
        {
            printf("Unable to open the HPC Schedule File ... quitting\n");
            exit(0);
        }

        fgets(buffer, 200, HPCScheduleFile);
        fgets(buffer, 200, HPCScheduleFile);
        l = buffer;
        t = strtok(l, "\t\n");
        isPer = atoi(t);
        t = strtok(l, "\t\n");
        numberOfItemsToSend = atoi(t);
        if (isPer == 1)
            numberOfItemsToSend = (numberOfItemsToSend * P_size) / 100;
        t = strtok(l, "\t\n");
        epochLength = atoi(t);
        t = strtok(l, "\t\n");
        BorW = atoi(t);
        t = strtok(l, "\t\n");
        fullHalt = atoi(t);
    }

    MPI_Bcast(&numberOfItemsToSend,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&epochLength,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&BorW,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&fullHalt,
              1,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    if (fullHalt == 0)
    {
#define FULLHALT
    }
}

#endif

/*************************************************************************************/
/*   Compares the fitness function of an individual against the population           */
/*  INPUT: get the Bin_Fullness value of the individual's num_item th attribute      */
/*  OUTPUT: Returns 0 if fitness is different, and 1 if there is an individual       */
/*          with the same fitness value. Does not necessarily be the same individual */
/*************************************************************************************/
int compare(double comparedBinFullness)
{
    int i;

    for (i = 0; i < P_size; i++)
    { // for each item in the population
        if (comparedBinFullness == population[i][number_items].Bin_Fullness)
        {
            return 1;
        }
    }
    return 0;
}

/***********************************************************************************/
/*   Compares the fitness function of an individual against the population         */
/* INPUT: gets a complete individual                                               */
/* OUTPUT: Returns 0 if the individual is different, and 1 if same. Note that the  */
/*         individual should have the exact same distribution on the items in Bins */
/***********************************************************************************/
int compare(SOLUTION *comparedPopulationItem)
{
    int i, j;
    int flag;

    for (i = 0; i < P_size; i++)
    { // for each item in the population
        if (comparedPopulationItem[number_items].Bin_Fullness == population[i][number_items].Bin_Fullness)
        {
            flag = 1;
            for (j = 0; j < number_items; j++)
            {
                if (comparedPopulationItem[j].Bin_Fullness != population[i][number_items].Bin_Fullness)
                {
                    flag = 0;
                }
            }
            if (flag)
            {
                return 1;
            }
        }
    }
    return 0;
}

void printAllPopulation(SOLUTION *Pop, FILE *out, int n_bins)
{
    int i;
    int bin, position;
    node *p;
    long int item;

    for (i = 0; i < P_size; i++)
    {
        fprintf(out, "POPULATION ITEM #%d\n", i);
        for (bin = 0; bin < n_bins; bin++)
        {
            fprintf(out, "BIN %d", bin);
            p = Pop[(i * P_size) + bin].L.first;
            while (p != NULL)
            {
                item = p->data;
                p = p->next;
                fprintf(out, "[Item: %ld, Weight: %ld]\t", item + 1, weight[item]);
            }
        }
    }
}

#ifdef MPI_ON

void Read_Schedule(char *fileName)
{

    int i, j, k;
    int readBuffer[1000];
    int readBufferSize;
    int sendBuffer[100000];
    int sendBufferSize;
    char buffer[1000];
    char *line, *line2, *token;
    node *temp;
    FILE *scheduleFile;

    ScheduleQueue allSendSchedules[size];
    ScheduleQueue allReceiveSchedules[size];

    if (myrank == 0)
    {

        scheduleFile = fopen(fileName, "r");
        if (!scheduleFile)
        {
            printf("Schedule File %s Does not exist, quitting\n", fileName);
            exit(0);
        }

        // printf("[%d] Starting\n", myrank);
        // MPI_Barrier(MPI_COMM_WORLD);

        fgets(buffer, 1000, scheduleFile);
        numScheduleSteps = atoi(buffer);

        for (i = 0; i < numScheduleSteps; i++)
        {
            fgets(buffer, 1000, scheduleFile);
            line = buffer;

            for (j = 0; j < size; j++)
            { // for each processor we need to have a list of schedule items
                token = strtok(line, ";");
                line2 = token;

                readBufferSize = 0;
                while (line2 != NULL && strlen(line2) != 0)
                {
                    token = strtok(line2, " \t");
                    // printf("|%s| -- |%s|\n\n", line2, token);
                    readBuffer[readBufferSize] = atoi(token);
                    readBufferSize++;

                    allSendSchedules[j].schedule[i] = (int *)malloc(readBufferSize * sizeof(int));
                    allSendSchedules[j].scheduleSizes[i] = readBufferSize;
                    for (k = 0; k < readBufferSize; k++)
                    {
                        allSendSchedules[j].schedule[i][k] = readBuffer[k];
                    }
                }
            }
        }

#ifdef VERBOSE_SCHEDULE
        for (i = 0; i < size; i++)
        {
            for (j = 0; j < numScheduleSteps; j++)
            {
                printf("AllSendSch[%d].ScheduleSizes[%d]: %d\n", i, j, allSendSchedules[i].scheduleSizes[j]);
                for (k = 0; k < allSendSchedules[i].scheduleSizes[j]; k++)
                {
                    printf("AllSendSch[%d].Schedule[%d][%d]: %d\n", i, j, k, allSendSchedules[i].schedule[j][k]);
                }
            }
        }
#endif

        // NOW WE HAVE EVERYONES SEND SCHEDULES ON ALLSENDSCHEDULES
        // WE NEED THE RECEIVE SCHEDULES TOO
        for (i = 0; i < numScheduleSteps; i++)
        {
            for (j = 0; j < size; j++)
            {
                allReceiveSchedules[j].schedule[i] = (int *)malloc(size * sizeof(int));
                allReceiveSchedules[j].scheduleSizes[i] = 0;
            }
        }

        for (i = 0; i < numScheduleSteps; i++)
        {
            for (j = 0; j < size; j++)
            {
                for (k = 0; k < allSendSchedules[j].scheduleSizes[i]; k++)
                {
                    allReceiveSchedules[allSendSchedules[j].schedule[i][k]].schedule[i][allReceiveSchedules[allSendSchedules[j].schedule[i][k]].scheduleSizes[i]] = j;
                    allReceiveSchedules[allSendSchedules[j].schedule[i][k]].scheduleSizes[i]++;
                }
            }
        }

#ifdef VERBOSE_SCHEDULE
        for (i = 0; i < size; i++)
        {
            for (j = 0; j < numScheduleSteps; j++)
            {
                for (k = 0; k < allReceiveSchedules[i].scheduleSizes[j]; k++)
                {
                    printf("AllRecvSch[%d].SS[%d][%d]: %d\n", i, j, k, allReceiveSchedules[i].schedule[j][k]);
                }
            }
        }
#endif

        // DONE - NOW WE CAN PROCEED COMMUNICATING THESE ARRAYS TO OTHER PROCESSORS
        // firstly number of schedule steps
        MPI_Bcast(&numScheduleSteps,
                  1,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);
        // printf("NSS[%d]:%d\n", myrank, numScheduleSteps);
        // MPI_Barrier(MPI_COMM_WORLD);

        // SENDING OF SEND SCHEDULES
        for (i = 0; i < size; i++)
        {
            for (j = 0; j < numScheduleSteps; j++)
            {
                sendBuffer[(i * numScheduleSteps + j)] = allSendSchedules[i].scheduleSizes[j];
                // printf("ASS[%d].SS[%d]: %d\n", i, j, allSendSchedules[i].scheduleSizes[j]);
            }
        }

        for (i = 0; i < numScheduleSteps; i++)
        {
            Send_Schedules.scheduleSizes[i] = allSendSchedules[0].scheduleSizes[i];
            Send_Schedules.schedule[i] = (int *)malloc(allSendSchedules[0].scheduleSizes[i] * sizeof(int));
            for (j = 0; j < Send_Schedules.scheduleSizes[i]; j++)
            {
                Send_Schedules.schedule[i][j] = allSendSchedules[0].schedule[i][j];
            }
        }

        for (i = 1; i < size; i++)
        {
            MPI_Send(&sendBuffer[(i * numScheduleSteps)],
                     numScheduleSteps,
                     MPI_INT,
                     i,
                     0,
                     MPI_COMM_WORLD);

            sendBufferSize = 0;
            for (j = 0; j < numScheduleSteps; j++)
            {
                for (k = 0; k < allSendSchedules[i].scheduleSizes[j]; k++)
                {
                    sendBuffer[sendBufferSize + k] = allSendSchedules[i].schedule[j][k];
                }
                sendBufferSize = sendBufferSize + allSendSchedules[i].scheduleSizes[j];
            }
            MPI_Send(&sendBuffer[0],
                     sendBufferSize,
                     MPI_INT,
                     i,
                     0,
                     MPI_COMM_WORLD);
        }

        // SENDING OF RECEIVE SCHEDULES
        for (i = 0; i < size; i++)
        {
            for (j = 0; j < numScheduleSteps; j++)
            {
                sendBuffer[(i * numScheduleSteps + j)] = allReceiveSchedules[i].scheduleSizes[j];
            }
        }
        for (i = 0; i < numScheduleSteps; i++)
        {
            Receive_Schedules.scheduleSizes[i] = allReceiveSchedules[0].scheduleSizes[i];
            Receive_Schedules.schedule[i] = (int *)malloc(allReceiveSchedules[0].scheduleSizes[i] * sizeof(int));
            for (j = 0; j < Receive_Schedules.scheduleSizes[i]; j++)
            {
                Receive_Schedules.schedule[i][j] = allReceiveSchedules[0].schedule[i][j];
            }
        }
        for (i = 1; i < size; i++)
        {
            MPI_Send(&sendBuffer[(i * numScheduleSteps)],
                     numScheduleSteps,
                     MPI_INT,
                     i,
                     0,
                     MPI_COMM_WORLD);

            sendBufferSize = 0;
            for (j = 0; j < numScheduleSteps; j++)
            {
                for (k = 0; k < allReceiveSchedules[i].scheduleSizes[j]; k++)
                {
                    sendBuffer[sendBufferSize + k] = allReceiveSchedules[i].schedule[j][k];
                }
                sendBufferSize = sendBufferSize + allReceiveSchedules[i].scheduleSizes[j];
            }
            MPI_Send(&sendBuffer[0],
                     sendBufferSize,
                     MPI_INT,
                     i,
                     0,
                     MPI_COMM_WORLD);
        }

        // DONE !!!!!!!
    }
    else
    { // if myrank != 0

        // printf("[%d] Starting\n", myrank);
        // MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&numScheduleSteps,
                  1,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);
        // printf("NSS[%d]:%d\n", myrank, numScheduleSteps);
        // MPI_Barrier(MPI_COMM_WORLD);

        MPI_Recv(&sendBuffer[0],
                 numScheduleSteps,
                 MPI_INT,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 &status);
        sendBufferSize = 0;
        for (i = 0; i < numScheduleSteps; i++)
        {
            Send_Schedules.scheduleSizes[i] = sendBuffer[i];
            sendBufferSize = sendBufferSize + sendBuffer[i];
        }

        MPI_Recv(&sendBuffer[0],
                 sendBufferSize,
                 MPI_INT,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 &status);

        k = 0;
        for (i = 0; i < numScheduleSteps; i++)
        {
            Send_Schedules.schedule[i] = (int *)malloc(Send_Schedules.scheduleSizes[i] * sizeof(int));
            for (j = 0; j < Send_Schedules.scheduleSizes[i]; j++)
            {
                Send_Schedules.schedule[i][j] = sendBuffer[k];
                k++;
            }
        }

#ifdef VERBOSE_SCHEDULE
        for (j = 0; j < numScheduleSteps; j++)
        {
            printf("[%d] - SendSch.ScheduleSizes[%d]: %d\n", myrank, j, Send_Schedules.scheduleSizes[j]);
            for (k = 0; k < Send_Schedules.scheduleSizes[j]; k++)
            {
                printf("[%d] - SendSch.SS[%d][%d]: %d\n", myrank, j, k, Send_Schedules.schedule[j][k]);
            }
        }
#endif

        MPI_Recv(&sendBuffer[0],
                 numScheduleSteps,
                 MPI_INT,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 &status);
        sendBufferSize = 0;
        for (i = 0; i < numScheduleSteps; i++)
        {
            Receive_Schedules.scheduleSizes[i] = sendBuffer[i];
            sendBufferSize = sendBufferSize + sendBuffer[i];
        }

        MPI_Recv(&sendBuffer[0],
                 sendBufferSize,
                 MPI_INT,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 &status);

        k = 0;
        for (i = 0; i < numScheduleSteps; i++)
        {
            Receive_Schedules.schedule[i] = (int *)malloc(Receive_Schedules.scheduleSizes[i] * sizeof(int));
            for (j = 0; j < Receive_Schedules.scheduleSizes[i]; j++)
            {
                Receive_Schedules.schedule[i][j] = sendBuffer[k];
                k++;
            }
        }

#ifdef VERBOSE_SCHEDULE
        for (j = 0; j < numScheduleSteps; j++)
        {
            printf("[%d] - RecvSch.scheduleSizes[%d]: %d\n", myrank, j, Receive_Schedules.scheduleSizes[j]);
            for (k = 0; k < Receive_Schedules.scheduleSizes[j]; k++)
            {
                printf("[%d] - RecvSch.SS[%d][%d]: %d\n", myrank, j, k, Receive_Schedules.schedule[j][k]);
            }
        }
#endif
        // DONE RECEIVING !!!!!!!
    }

    // Before Finishing
    // Everyone needs to calculate the maximum number of receives they will do
    // in a single schedule step
    // this will be used in data communication
    // in order not to waste any space.
    k = 0;
    for (i = 0; i < numScheduleSteps; i++)
    {
        if (k < Receive_Schedules.scheduleSizes[i])
        {
            k = Receive_Schedules.scheduleSizes[i];
        }
    }
    MaxCommsPerScheduleStep = k;

    // printf("SCHEDULE TRANSFER DONE[%d]\n", myrank);

} // END OF READ SCHEDULE

void initQueue(ScheduleQueue *queue, int size)
{
    queue->size = size;
    queue->start = 0;
    queue->end = size;
}

int getCurrentQueueItemCount(ScheduleQueue *queue)
{
    return (queue->scheduleSizes[queue->start]);
}

int *getCurrentQueueItems(ScheduleQueue *queue)
{
    return (queue->schedule[queue->start]);
}

void queueNext(ScheduleQueue *queue)
{
    queue->end = ((queue->end + 1) % queue->size);
    queue->start = (queue->start + 1) % queue->size;
}
#endif

/************************************************************************************************************************
 To generate an initial population P of individuals with FF-� packing heuristic.                                        *
  population[i][number_items].Bin_Fullness: Saves the fitness of the solution i                                         *
  population[i][number_items + 1].Bin_Fullness:                                                                         *
    Saves the total number of bins in the solution i                                                                    *
  population[i][number_items + 2].Bin_Fullness:                                                                         *
    Saves the generation in which the solution i was generated                                                          *
  population[i][number_items + 3].Bin_Fullness:                                                                         *
    Saves the number of bins in the solution i that are fully at 100%                                                   *
  population[i][number_items + 4].Bin_Fullness:                                                                         *
    Saves the fullness of the bin with the highest avaliable capacity in the solution i                                 *
 Output:                                                                                                                *
    (1) when it finds a solution for which the size matches the L2 lower bound                                          *
    (0) otherwise                                                                                                       *
************************************************************************************************************************/
long int Generate_Initial_Population()
{
    for (i = 0; i < P_size; i++)
    {
        FF_n_(i);
        population[i][number_items + 2].Bin_Fullness = generation;
        population[i][number_items].Bin_Fullness /= population[i][number_items + 1].Bin_Fullness;
        if (population[i][number_items + 1].Bin_Fullness == L2)
        {
            clock_t end = clock();
            Copy_Solution(global_best_solution, population[i], 0);
            global_best_solution[number_items].Bin_Fullness = population[i][number_items].Bin_Fullness;
            global_best_solution[number_items + 2].Bin_Fullness = generation;
            global_best_solution[number_items + 1].Bin_Fullness = population[i][number_items + 1].Bin_Fullness;
            global_best_solution[number_items + 3].Bin_Fullness = population[i][number_items + 3].Bin_Fullness;
            TotalTime = (end - start) / (1000 * 1.0);
            WriteOutput();
            is_optimal_solution = 1;
            return (1);
        }
    }
    return 0;
}

/************************************************************************************************************************
 To apply the reproduction technique: Controlled selection and Controlled replacement.                                          *
 Output:                                                                                                                        *
    (1) when it finds a solution for which the size matches the L2 lower bound                                                      *
    (2) if more than 0.1*P_size individuals (solutions) have duplicated-fitness                                                      *
    (0) otherwise                                                                                                                                            *
************************************************************************************************************************/
long int Generation()
{
    long int f1;
    long int f2;
    long int h;
    long int k;

    /*-----------------------------------------------------------------------------------------------------
    ---------------------------------Controlled selection for crossover------------------------------------
    -----------------------------------------------------------------------------------------------------*/
    Sort_Ascending_IndividualsFitness();
    if (generation > 1 && repeated_fitness > 0.1 * P_size)
        return (2);
    Sort_Random(random_individuals, 0, (int)(P_size - (int)(P_size * B_size)));
    Sort_Random(best_individuals, (int)((1 - p_c) * P_size), P_size);
    k = 0;
    h = P_size - 1;

    for (i = P_size - 1, j = 0; i > P_size - (p_c / 2 * P_size); i--, j += 2)
    {
        f1 = best_individuals[h--];
        f2 = random_individuals[k++];
        if (f2 == f1)
        {
            f1 = best_individuals[h--];
        }
        Gene_Level_Crossover_FFD(ordered_population[f1], ordered_population[f2], j);
        children[j][number_items + 2].Bin_Fullness = generation + 1;
        children[j][number_items].Bin_Fullness /= children[j][number_items + 1].Bin_Fullness;
        if (children[j][number_items + 1].Bin_Fullness == L2)
        {
            clock_t end = clock();
            Copy_Solution(global_best_solution, children[j], 0);
            global_best_solution[number_items].Bin_Fullness = children[j][number_items].Bin_Fullness;
            ;
            global_best_solution[number_items + 2].Bin_Fullness = generation + 1;
            global_best_solution[number_items + 1].Bin_Fullness = children[j][number_items + 1].Bin_Fullness;
            global_best_solution[number_items + 3].Bin_Fullness = children[j][number_items + 3].Bin_Fullness;
            TotalTime = (end - start) / (1000 * 1.0);
            WriteOutput();
            is_optimal_solution = 1;
            return (1);
        }
        Gene_Level_Crossover_FFD(ordered_population[f2], ordered_population[f1], j + 1);
        children[j + 1][number_items + 2].Bin_Fullness = generation + 1;
        children[j + 1][number_items].Bin_Fullness /= children[j + 1][number_items + 1].Bin_Fullness;
        if (children[j + 1][number_items + 1].Bin_Fullness == L2)
        {
            clock_t end = clock();
            Copy_Solution(global_best_solution, children[j + 1], 0);
            global_best_solution[number_items].Bin_Fullness = children[j + 1][number_items].Bin_Fullness;
            ;
            global_best_solution[number_items + 2].Bin_Fullness = generation + 1;
            global_best_solution[number_items + 1].Bin_Fullness = children[j + 1][number_items + 1].Bin_Fullness;
            global_best_solution[number_items + 3].Bin_Fullness = children[j + 1][number_items + 3].Bin_Fullness;
            TotalTime = (end - start) / (1000 * 1.0);
            WriteOutput();
            is_optimal_solution = 1;
            return (1);
        }
    }

    /*-----------------------------------------------------------------------------------------------------
    ---------------------------------Controlled replacement for crossover----------------------------------
    -----------------------------------------------------------------------------------------------------*/
    k = 0;
    for (j = 0; j < p_c / 2 * P_size - 1; j++)
        Copy_Solution(population[ordered_population[random_individuals[k++]]], children[j], 1);

    k = 0;
    for (i = P_size - 1; i > P_size - (p_c / 2 * P_size); i--, j++)
    {
        while (population[ordered_population[k]][number_items + 2].Bin_Fullness == generation + 1)
            k++;
        Copy_Solution(population[ordered_population[k++]], children[j], 1);
    }
    /*-----------------------------------------------------------------------------------------------------
    --------------------------------Controlled selection for mutation--------------------------------------
    -----------------------------------------------------------------------------------------------------*/
    Sort_Ascending_IndividualsFitness();
    if (generation > 1 && repeated_fitness > 0.1 * P_size)
        return (2);
    j = 0;
    for (i = P_size - 1; i > P_size - (p_m * P_size); i--)
    {
        if (j < (int)(P_size * B_size) && generation + 1 - population[ordered_population[i]][number_items + 2].Bin_Fullness < life_span)
        {
            /*-----------------------------------------------------------------------------------------------------
            ----------------------------------Controlled replacement for mutation----------------------------------
            -----------------------------------------------------------------------------------------------------*/
            Copy_Solution(population[ordered_population[j]], population[ordered_population[i]], 0);
            Adaptive_Mutation_RP(ordered_population[j], k_cs, 1);
            population[ordered_population[j]][number_items + 2].Bin_Fullness = generation + 1;
            population[ordered_population[j]][number_items].Bin_Fullness /= population[ordered_population[j]][number_items + 1].Bin_Fullness;
            if (population[ordered_population[j]][number_items + 1].Bin_Fullness == L2)
            {
                clock_t end = clock();
                Copy_Solution(global_best_solution, population[ordered_population[j]], 0);
                global_best_solution[number_items].Bin_Fullness = population[ordered_population[j]][number_items].Bin_Fullness;
                ;
                global_best_solution[number_items + 2].Bin_Fullness = generation + 1;
                global_best_solution[number_items + 1].Bin_Fullness = population[ordered_population[j]][number_items + 1].Bin_Fullness;
                global_best_solution[number_items + 3].Bin_Fullness = population[ordered_population[j]][number_items + 3].Bin_Fullness;
                TotalTime = (end - start) / (1000 * 1.0);
                WriteOutput();
                is_optimal_solution = 1;
                return (1);
            }
            j++;
        }
        else
        {
            Adaptive_Mutation_RP(ordered_population[i], k_ncs, 0);
            population[ordered_population[i]][number_items + 2].Bin_Fullness = generation + 1;
            population[ordered_population[i]][number_items].Bin_Fullness /= population[ordered_population[i]][number_items + 1].Bin_Fullness;
            if (population[ordered_population[i]][number_items + 1].Bin_Fullness == L2)
            {
                clock_t end = clock();
                Copy_Solution(global_best_solution, population[ordered_population[i]], 0);
                global_best_solution[number_items].Bin_Fullness = population[ordered_population[i]][number_items].Bin_Fullness;
                ;
                global_best_solution[number_items + 2].Bin_Fullness = generation + 1;
                global_best_solution[number_items + 1].Bin_Fullness = population[ordered_population[i]][number_items + 1].Bin_Fullness;
                global_best_solution[number_items + 3].Bin_Fullness = population[ordered_population[i]][number_items + 3].Bin_Fullness;
                TotalTime = (end - start) / (1000 * 1.0);
                WriteOutput();
                is_optimal_solution = 1;
                return (1);
            }
        }
    }
    return 0;
}

/************************************************************************************************************************
 To recombine two parent solutions producing a child solution.                                                          *
 Input:                                                                                                                 *
    The positions in the population of the two parent solutions: father_1 and father_2                                  *
    The position in the set of children of the child solution: child                                                    *
************************************************************************************************************************/
void Gene_Level_Crossover_FFD(long int father_1, long int father_2, long int child)
{
    long int k;
    long int counter;
    long int k2 = 0;
    long int ban = 1;
    long int items[ATTRIBUTES] = {0};
    long int free_items[ATTRIBUTES] = {0};

    children[child][number_items + 4].Bin_Fullness = bin_capacity;

    if (population[father_1][number_items + 1].Bin_Fullness > population[father_2][number_items + 1].Bin_Fullness)
        counter = (long int)population[father_1][number_items + 1].Bin_Fullness;
    else
        counter = (long int)population[father_2][number_items + 1].Bin_Fullness;

    long int *random_order1 = new long int[counter];
    long int *random_order2 = new long int[counter];

    for (k = 0; k < counter; k++)
    {
        random_order1[k] = k;
        random_order2[k] = k;
    }

    Sort_Random(random_order1, 0, (long int)population[father_1][number_items + 1].Bin_Fullness);
    Sort_Random(random_order2, 0, (long int)population[father_2][number_items + 1].Bin_Fullness);
    Sort_Descending_BinFullness(random_order1, father_1);
    Sort_Descending_BinFullness(random_order2, father_2);

    for (k = 0; k < population[father_1][number_items + 1].Bin_Fullness; k++)
    {
        if (population[father_1][random_order1[k]].Bin_Fullness >= population[father_2][random_order2[k]].Bin_Fullness)
        {
            ban = Used_Items(father_1, random_order1[k], items);
            if (ban == 1)
            {
                children[child][k2].L.clone_linked_list(population[father_1][random_order1[k]].L);
                children[child][k2++].Bin_Fullness = population[father_1][random_order1[k]].Bin_Fullness;
                if (children[child][k2 - 1].Bin_Fullness < children[child][number_items + 4].Bin_Fullness)
                    children[child][number_items + 4].Bin_Fullness = children[child][k2 - 1].Bin_Fullness;
            }
            if (population[father_2][random_order2[k]].Bin_Fullness > 0)
            {
                ban = Used_Items(father_2, random_order2[k], items);
                if (ban == 1)
                {
                    children[child][k2].L.clone_linked_list(population[father_2][random_order2[k]].L);
                    children[child][k2++].Bin_Fullness = population[father_2][random_order2[k]].Bin_Fullness;
                    if (children[child][k2 - 1].Bin_Fullness < children[child][number_items + 4].Bin_Fullness)
                        children[child][number_items + 4].Bin_Fullness = children[child][k2 - 1].Bin_Fullness;
                }
            }
        }
        else
        {
            if (population[father_2][random_order2[k]].Bin_Fullness > 0)
            {
                ban = Used_Items(father_2, random_order2[k], items);
                if (ban == 1)
                {
                    children[child][k2].L.clone_linked_list(population[father_2][random_order2[k]].L);
                    children[child][k2++].Bin_Fullness = population[father_2][random_order2[k]].Bin_Fullness;
                    if (children[child][k2 - 1].Bin_Fullness < children[child][number_items + 4].Bin_Fullness)
                        children[child][number_items + 4].Bin_Fullness = children[child][k2 - 1].Bin_Fullness;
                }
            }
            ban = Used_Items(father_1, random_order1[k], items);
            if (ban == 1)
            {
                children[child][k2].L.clone_linked_list(population[father_1][random_order1[k]].L);
                children[child][k2++].Bin_Fullness = population[father_1][random_order1[k]].Bin_Fullness;
                if (children[child][k2 - 1].Bin_Fullness < children[child][number_items + 4].Bin_Fullness)
                    children[child][number_items + 4].Bin_Fullness = children[child][k2 - 1].Bin_Fullness;
            }
        }
    }

    k = 0;
    for (counter = 0; counter < number_items; counter++)
    {
        if (items[ordered_weight[counter]] == 0)
            free_items[k++] = ordered_weight[counter];
    }

    if (k > 0)
    {
        bin_i = 0;
        for (counter = 0; counter < k - 1; counter++)
            FF(free_items[counter], children[child], k2, bin_i, 0);
        FF(free_items[counter], children[child], k2, bin_i, 1);
    }
    else
        for (k = 0; k < k2; k++)
            children[child][number_items].Bin_Fullness += pow((children[child][k].Bin_Fullness / bin_capacity), 2);
    children[child][number_items + 1].Bin_Fullness = k2;
    free(random_order1);
    free(random_order2);
}

/************************************************************************************************************************
 To produce a small modification in a solution.                                                                         *
 Input:                                                                                                                 *
    The position in the population of the solution to mutate: individual                                                *
    The rate of change to calculate the number of bins to eliminate: k                                                  *
    A value that indicates if the solution was cloned: is_cloned                                                        *
************************************************************************************************************************/
void Adaptive_Mutation_RP(long int individual, float k, int is_cloned)
{
    long int number_bins;
    long int i;
    long int lightest_bin = 0;
    long int number_free_items = 0;
    long int free_items[ATTRIBUTES] = {0};
    long int ordered_BinFullness[ATTRIBUTES] = {0};

    node *p;

    for (i = 0; i < population[individual][number_items + 1].Bin_Fullness; i++)
        ordered_BinFullness[i] = i;
    if (is_cloned)
        Sort_Random(ordered_BinFullness, 0, (long int)population[individual][number_items + 1].Bin_Fullness);
    Sort_Ascending_BinFullness(ordered_BinFullness, individual);

    i = 1;
    while (population[individual][ordered_BinFullness[i]].Bin_Fullness < bin_capacity && i < population[individual][number_items + 1].Bin_Fullness)
        i++;
    _p_ = 1 / (float)(k);

    number_bins = (long int)ceil(i * ((2 - i / population[individual][number_items + 1].Bin_Fullness) / pow(i, _p_)) *
                                 (1 - ((double)get_rand(&seed_emptybin, (long int)ceil((1 / pow(i, _p_)) * 100)) / 100)));

    for (i = 0; i < number_bins; i++)
    {
        p = population[individual][ordered_BinFullness[lightest_bin]].L.first;
        while (p != NULL)
        {
            free_items[number_free_items++] = p->data;
            p = p->next;
        }
        population[individual][ordered_BinFullness[lightest_bin]].L.free_linked_list();
        population[individual][ordered_BinFullness[lightest_bin]].Bin_Fullness = 0;
        lightest_bin++;
    }
    population[individual][number_items + 1].Bin_Fullness -= number_bins;
    number_bins = (long int)population[individual][number_items + 1].Bin_Fullness;
    Adjust_Solution(individual);
    RP(individual, number_bins, free_items, number_free_items);
    population[individual][number_items + 1].Bin_Fullness = number_bins;
}

/************************************************************************************************************************
 To generate a random BPP solution with the � large items packed in separate bins.                                      *
 Input:                                                                                                                 *
    The position in the population of the new solution: individual                                                      *
************************************************************************************************************************/
void FF_n_(int individual)
{
    long int i, j = 0, total_bins = 0;

    bin_i = 0;
    population[individual][number_items + 3].Bin_Fullness = 0;
    population[individual][number_items + 4].Bin_Fullness = bin_capacity;
    if (n_ > 0)
    {
        for (i = 0; i < n_; i++)
        {
            population[individual][i].Bin_Fullness = weight[ordered_weight[i]];
            population[individual][i].L.insert(ordered_weight[i]);
            total_bins++;
            if (population[individual][i].Bin_Fullness < population[individual][number_items + 4].Bin_Fullness)
                population[individual][number_items + 4].Bin_Fullness = population[individual][i].Bin_Fullness;
        }
        i = number_items - i;
        Sort_Random(permutation, 0, i);
        for (j = 0; j < i - 1; j++)
            FF(permutation[j], population[individual], total_bins, bin_i, 0);
        FF(permutation[j], population[individual], total_bins, bin_i, 1);
    }
    else
    {
        Sort_Random(permutation, 0, number_items);
        for (j = 0; j < number_items - 1; j++)
            FF(permutation[j], population[individual], total_bins, bin_i, 0);
        FF(permutation[j], population[individual], total_bins, bin_i, 1);
    }
    population[individual][number_items + 1].Bin_Fullness = total_bins;
}

/************************************************************************************************************************
 To reinsert free items into an incomplete BPP solution.                                                                *
 Input:                                                                                                                 *
    The position in the population of the incomplete solution where the free items must be reinserted: individual       *
   The number of bins of the partial_solution: b                                                                        *
   A set of free items to be reinserted into the partial_solution: F                                                    *
   The number of free items of F: number_free_items                                                                     *
************************************************************************************************************************/
void RP(long int individual, long int &b, long int F[], long int number_free_items)
{
    long int i;
    long int k;
    long int k2;
    long int ban;
    long int sum = 0;
    long int total_free = 0;
    long int ordered_BinFullness[ATTRIBUTES] = {0};
    long int *new_free_items = new long int[2];

    node *p, *s, *aux;

    higher_weight = weight[F[0]];
    lighter_weight = weight[F[0]];
    bin_i = b;
    population[individual][number_items].Bin_Fullness = 0;
    population[individual][number_items + 3].Bin_Fullness = 0;
    population[individual][number_items + 4].Bin_Fullness = bin_capacity;

    for (i = 0; i < b; i++)
        ordered_BinFullness[i] = i;
    Sort_Random(ordered_BinFullness, 0, b);
    Sort_Random(F, 0, number_free_items);

    for (i = 0; i < b; i++)
    {
        sum = (long int)population[individual][ordered_BinFullness[i]].Bin_Fullness;
        p = population[individual][ordered_BinFullness[i]].L.first;
        while (p->next != NULL)
        {
            ban = 0;
            aux = p;
            s = p->next;
            while (s != NULL)
            {
                for (k = 0; k < number_free_items - 1; k++)
                {
                    if (i == b - 1)
                        if (weight[F[k]] > higher_weight)
                            higher_weight = weight[F[k]];
                    for (k2 = k + 1; k2 < number_free_items; k2++)
                    {
                        if (weight[F[k]] >= weight[p->data] + weight[s->data] && ((sum - (weight[p->data] + weight[s->data]) + (weight[F[k]]) <= bin_capacity)))
                        {
                            sum = sum - (weight[p->data] + weight[s->data]) + (weight[F[k]]);
                            new_free_items[0] = p->data;
                            new_free_items[1] = s->data;
                            p->data = F[k];
                            aux->next = s->next;
                            free(s);
                            if (population[individual][ordered_BinFullness[i]].L.last == s)
                                population[individual][ordered_BinFullness[i]].L.last = aux;
                            population[individual][ordered_BinFullness[i]].L.num--;
                            F[k] = new_free_items[0];
                            F[number_free_items + total_free] = new_free_items[1];
                            total_free++;
                            ban = 1;
                            break;
                        }
                        if (weight[F[k2]] >= weight[p->data] + weight[s->data] && ((sum - (weight[p->data] + weight[s->data]) + (weight[F[k2]]) <= bin_capacity)))
                        {
                            sum = sum - (weight[p->data] + weight[s->data]) + (weight[F[k2]]);
                            new_free_items[0] = p->data;
                            new_free_items[1] = s->data;
                            p->data = F[k2];
                            aux->next = s->next;
                            free(s);
                            if (population[individual][ordered_BinFullness[i]].L.last == s)
                                population[individual][ordered_BinFullness[i]].L.last = aux;
                            population[individual][ordered_BinFullness[i]].L.num--;
                            F[k2] = new_free_items[0];
                            F[number_free_items + total_free] = new_free_items[1];
                            total_free++;
                            ban = 1;
                            break;
                        }
                        if ((weight[F[k]] + weight[F[k2]] > weight[p->data] + weight[s->data]) ||
                            ((weight[F[k]] + weight[F[k2]] == weight[p->data] + weight[s->data]) && !(weight[F[k]] == weight[p->data] || weight[F[k]] == weight[s->data])))
                        {
                            if (sum - (weight[p->data] + weight[s->data]) + (weight[F[k]] + weight[F[k2]]) > bin_capacity)
                                break;
                            sum = sum - (weight[p->data] + weight[s->data]) + (weight[F[k]] + weight[F[k2]]);
                            new_free_items[0] = p->data;
                            new_free_items[1] = s->data;
                            p->data = F[k];
                            s->data = F[k2];
                            F[k] = new_free_items[0];
                            F[k2] = new_free_items[1];
                            if (sum == bin_capacity)
                            {
                                ban = 1;
                                break;
                            }
                        }
                    }
                    if (ban)
                        break;
                }
                if (ban)
                    break;
                aux = s;
                s = s->next;
            }
            if (ban)
                break;
            p = p->next;
        }
        population[individual][ordered_BinFullness[i]].Bin_Fullness = sum;
        if (population[individual][ordered_BinFullness[i]].Bin_Fullness < population[individual][number_items + 4].Bin_Fullness)
            population[individual][number_items + 4].Bin_Fullness = population[individual][ordered_BinFullness[i]].Bin_Fullness;
        if (population[individual][ordered_BinFullness[i]].Bin_Fullness == bin_capacity)
            population[individual][number_items + 3].Bin_Fullness++;
        else if (population[individual][ordered_BinFullness[i]].Bin_Fullness + weight[ordered_weight[number_items - 1]] <= bin_capacity)
        {
            if (ordered_BinFullness[i] < bin_i)
                bin_i = ordered_BinFullness[i];
        }
    }
    for (i = 0; i < bin_i; i++)
        population[individual][number_items].Bin_Fullness += pow((population[individual][i].Bin_Fullness / bin_capacity), 2);

    free(new_free_items);

    number_free_items += total_free;

    if (higher_weight < .5 * bin_capacity)
        Sort_Random(F, 0, number_free_items);
    else
    {
        Sort_Descending_Weights(F, number_free_items);
        lighter_weight = weight[F[number_free_items - 1]];
    }

    if (lighter_weight > bin_capacity - population[individual][number_items + 4].Bin_Fullness)
    {
        for (i = bin_i; i < b; i++)
            population[individual][number_items].Bin_Fullness += pow((population[individual][i].Bin_Fullness / bin_capacity), 2);
        bin_i = b;
    }
    for (i = 0; i < number_free_items - 1; i++)
        FF(F[i], population[individual], b, bin_i, 0);
    FF(F[i], population[individual], b, bin_i, 1);
}

/************************************************************************************************************************
 To insert an item into an incomplete BPP solution.                                                                     *
 Input:                                                                                                                 *
   An item to be inserted into the individual: item                                                                     *
    An incomplete chromosome where the item must be inserted: individual                                                *
   The number of bins of the individual: total_bins                                                                     *
   The first bin that could have sufficient available capacity to store the item: beginning                             *
   A value that indicates if it is the last item to be stored into the individual: is_last                              *
************************************************************************************************************************/
void FF(long int item, SOLUTION individual[], long int &total_bins, long int beginning, int is_last)
{
    long int i;

    if (!is_last && weight[item] > (bin_capacity - (long int)individual[number_items + 4].Bin_Fullness))
        i = total_bins;
    else
        for (i = beginning; i < total_bins; i++)
        {
            if ((long int)individual[i].Bin_Fullness + weight[item] <= bin_capacity)
            {
                individual[i].Bin_Fullness += weight[item];
                individual[i].L.insert(item);
                if ((long int)individual[i].Bin_Fullness == bin_capacity)
                    individual[number_items + 3].Bin_Fullness++;
                if (is_last)
                {
                    for (i; i < total_bins; i++)
                        individual[number_items].Bin_Fullness += pow((individual[i].Bin_Fullness / bin_capacity), 2);
                    return;
                }
                if ((long int)individual[i].Bin_Fullness + weight[ordered_weight[number_items - 1]] > bin_capacity && i == bin_i)
                {
                    bin_i++;
                    individual[number_items].Bin_Fullness += pow((individual[i].Bin_Fullness / bin_capacity), 2);
                }
                return;
            }
            if (is_last)
                individual[number_items].Bin_Fullness += pow((individual[i].Bin_Fullness / bin_capacity), 2);
        }
    individual[i].Bin_Fullness += weight[item];
    individual[i].L.insert(item);
    if (individual[i].Bin_Fullness < individual[number_items + 4].Bin_Fullness)
        individual[number_items + 4].Bin_Fullness = individual[i].Bin_Fullness;
    if (is_last)
        individual[number_items].Bin_Fullness += pow((individual[i].Bin_Fullness / bin_capacity), 2);
    total_bins++;
}

/************************************************************************************************************************
 To calculate the lower bound L2 of Martello and Toth and the � large items n_                                          *
************************************************************************************************************************/
void LowerBound()
{
    long int k, m, i, j, aux1, aux2;
    long double sjx = 0, sj2 = 0, sj3 = 0;
    long int jx = 0, cj12, jp = 0, jpp = 0, cj2;

    while (weight[ordered_weight[jx]] > bin_capacity / 2 && jx < number_items)
        jx++;
    n_ = jx;
    if (jx == number_items)
    {
        L2 = jx;
        return;
    }
    if (jx == 0)
    {
        if (fmod(total_accumulated_weight, bin_capacity) >= 1)
            L2 = (long int)ceil(total_accumulated_weight / bin_capacity);
        else
            L2 = (long int)(total_accumulated_weight / bin_capacity);
        return;
    }
    else
    {
        cj12 = jx;
        for (i = jx; i < number_items; i++)
            sjx += weight[ordered_weight[i]];
        jp = jx;
        for (i = 0; i < jx; i++)
        {
            if (weight[ordered_weight[i]] <= bin_capacity - weight[ordered_weight[jx]])
            {
                jp = i;
                break;
            }
        }

        cj2 = jx - jp;
        for (i = jp; i <= jx - 1; i++)
            sj2 += weight[ordered_weight[i]];
        jpp = jx;
        sj3 = weight[ordered_weight[jpp]];
        ordered_weight[number_items] = number_items;
        weight[number_items] = 0;
        while (weight[ordered_weight[jpp + 1]] == weight[ordered_weight[jpp]])
        {
            jpp++;
            sj3 += weight[ordered_weight[jpp]];
        }
        L2 = cj12;

        do
        {
            if (fmod((sj3 + sj2), bin_capacity) >= 1)
                aux1 = (long int)ceil((sj3 + sj2) / bin_capacity - cj2);
            else
                aux1 = (long int)((sj3 + sj2) / bin_capacity - cj2);

            if (L2 < (cj12 + aux1))
                L2 = cj12 + aux1;
            jpp++;
            if (jpp < number_items)
            {
                sj3 += weight[ordered_weight[jpp]];
                while (weight[ordered_weight[jpp + 1]] == weight[ordered_weight[jpp]])
                {
                    jpp++;
                    sj3 += weight[ordered_weight[jpp]];
                }
                while (jp > 0 && weight[ordered_weight[jp - 1]] <= bin_capacity - weight[ordered_weight[jpp]])
                {
                    jp--;
                    cj2++;
                    sj2 += weight[ordered_weight[jp]];
                }
            }
            if (fmod((sjx + sj2), bin_capacity) >= 1)
                aux2 = (long int)ceil((sjx + sj2) / bin_capacity - cj2);
            else
                aux2 = (long int)((sjx + sj2) / bin_capacity - cj2);
        } while (jpp <= number_items || (cj12 + aux2) > L2);
    }
}

/************************************************************************************************************************
 To find the solution with the highest fitness of the population and update the global_best_solution                    *
************************************************************************************************************************/
void Find_Best_Solution()
{
    long int i,
        best_individual = 0;
    for (i = 0; i < P_size; i++)
    {
        if (population[i][number_items].Bin_Fullness > population[best_individual][number_items].Bin_Fullness)
            best_individual = i;
    }

    if (generation > 0)
    {
        if (population[best_individual][number_items].Bin_Fullness > global_best_solution[number_items].Bin_Fullness)
        {
            Copy_Solution(global_best_solution, population[best_individual], 0);
        }
    }
    else
    {
        Copy_Solution(global_best_solution, population[best_individual], 0);
    }
}

/************************************************************************************************************************
 To sort the individuals of the population in ascending order of their fitness                                          *
************************************************************************************************************************/
void Sort_Ascending_IndividualsFitness()
{
    long int i;
    long int k = P_size - 1;
    long int i2 = 0;
    long int aux;
    long int ban = 1;

    while (ban)
    {
        ban = 0;
        for (i = i2; i < k; i++)
        {
            if (population[ordered_population[i]][number_items].Bin_Fullness > population[ordered_population[i + 1]][number_items].Bin_Fullness)
            {
                aux = ordered_population[i];
                ordered_population[i] = ordered_population[i + 1];
                ordered_population[i + 1] = aux;
                ban = 1;
            }
            else if (population[ordered_population[i]][number_items].Bin_Fullness == population[ordered_population[i + 1]][number_items].Bin_Fullness)
            {
                aux = ordered_population[i + 1];
                ordered_population[i + 1] = ordered_population[i2];
                ordered_population[i2] = aux;
                i2++;
            }
        }
        k--;
    }
    repeated_fitness = i2;
}

/************************************************************************************************************************
 To sort the bins of a solution in ascending order of their filling                                                     *
 Input:                                                                                                                 *
    An array to save the order of the bins: ordered_BinFullness                                                         *                                                                                                                   *
    The position in the population of the solution: individual                                                          *
************************************************************************************************************************/
void Sort_Ascending_BinFullness(long int ordered_BinFullness[], long int individual)
{
    long int m, k;
    long int temporary_variable;
    long int ban = 1;

    k = (long int)population[individual][number_items + 1].Bin_Fullness - 1;
    while (ban)
    {
        ban = 0;
        for (m = 0; m < k; m++)
        {
            if (population[individual][ordered_BinFullness[m]].Bin_Fullness > population[individual][ordered_BinFullness[m + 1]].Bin_Fullness)
            {
                temporary_variable = ordered_BinFullness[m];
                ordered_BinFullness[m] = ordered_BinFullness[m + 1];
                ordered_BinFullness[m + 1] = temporary_variable;
                ban = 1;
            }
        }
        k--;
    }
}

/************************************************************************************************************************
 To sort the bins of a solution in descending order of their filling                                                    *
 Input:                                                                                                                 *
    An array to save the order of the bins: ordered_BinFullness                                                         *                                                                                                                   *
    The position in the population of the solution: individual                                                          *
************************************************************************************************************************/
void Sort_Descending_BinFullness(long int ordered_BinFullness[], long int individual)
{
    long int m, k;
    long int temporary_variable;
    long int ban = 1;

    k = (long int)population[individual][number_items + 1].Bin_Fullness - 1;
    while (ban)
    {
        ban = 0;
        for (m = 0; m < k; m++)
        {
            if (population[individual][ordered_BinFullness[m]].Bin_Fullness < population[individual][ordered_BinFullness[m + 1]].Bin_Fullness)
            {
                temporary_variable = ordered_BinFullness[m];
                ordered_BinFullness[m] = ordered_BinFullness[m + 1];
                ordered_BinFullness[m + 1] = temporary_variable;
                ban = 1;
            }
        }
        k--;
    }
}

/************************************************************************************************************************
 To sort the elements between the positions [k] and [n] of an array in random order                                     *
 Input:                                                                                                                 *
    The array to be randomized: random_array                                                                            *                                                                                                                   *
    The initial random position: k                                                                                      *
    The final random position: n                                                                                        *
************************************************************************************************************************/
void Sort_Random(long int random_array[], long int k, int n)
{

    long int i;
    long int aux;
    long int random_number;

    for (i = n - 1; i >= k; i--)
    {
        random_number = k + get_rand(&seed_permutation, n - k) - 1;
        aux = random_array[random_number];
        random_array[random_number] = random_array[i];
        random_array[i] = aux;
        if (weight[random_array[i]] < lighter_weight)
            lighter_weight = weight[random_array[i]];
        if (weight[random_array[random_number]] < lighter_weight)
            lighter_weight = weight[random_array[random_number]];
    }
}

/************************************************************************************************************************
 To sort a set of items in descending order of their weights                                                            *
 Input:                                                                                                                 *
    An array to save the order of the items: ordered_weight                                                             *                                                                                                                   *
    The number of items in the set: n                                                                                   *
************************************************************************************************************************/
void Sort_Descending_Weights(long int ordered_weight[], long int n)
{
    long int m, k;
    long int temporary_variable;
    long int ban = 1;

    k = n - 1;
    while (ban)
    {
        ban = 0;
        for (m = 0; m < k; m++)
        {
            if (weight[ordered_weight[m]] < weight[ordered_weight[m + 1]])
            {
                temporary_variable = ordered_weight[m];
                ordered_weight[m] = ordered_weight[m + 1];
                ordered_weight[m + 1] = temporary_variable;
                ban = 1;
            }
        }
        k--;
    }
}

/************************************************************************************************************************
 To copy solution2 in solution                                                                                          *
 Input:                                                                                                                 *
    A solution to save the copied solution: solution                                                                    *                                                                                                                   *
    The solution to be copied: solution2                                                                                *
    A value that indicates if the copied solution must be deleted: delete_solution2                                     *
************************************************************************************************************************/
void Copy_Solution(SOLUTION solution[], SOLUTION solution2[], int delete_solution2)
{

    long int j;

    for (j = 0; j < solution2[number_items + 1].Bin_Fullness; j++)
    {
        solution[j].Bin_Fullness = solution2[j].Bin_Fullness;
        solution[j].L.clone_linked_list(solution2[j].L);
        if (delete_solution2)
        {
            solution2[j].Bin_Fullness = 0;
            solution2[j].L.free_linked_list();
        }
    }
    while (j < solution[number_items + 1].Bin_Fullness)
    {
        solution[j].Bin_Fullness = 0;
        solution[j++].L.free_linked_list();
    }
    solution[number_items].Bin_Fullness = solution2[number_items].Bin_Fullness;
    solution[number_items + 1].Bin_Fullness = solution2[number_items + 1].Bin_Fullness;
    solution[number_items + 2].Bin_Fullness = solution2[number_items + 2].Bin_Fullness;
    solution[number_items + 3].Bin_Fullness = solution2[number_items + 3].Bin_Fullness;
    solution[number_items + 4].Bin_Fullness = solution2[number_items + 4].Bin_Fullness;
    if (delete_solution2)
    {
        solution2[number_items].Bin_Fullness = 0;
        solution2[number_items + 1].Bin_Fullness = 0;
        solution2[number_items + 2].Bin_Fullness = 0;
        solution2[number_items + 3].Bin_Fullness = 0;
        solution2[number_items + 4].Bin_Fullness = 0;
    }
}

/************************************************************************************************************************
 To free the memory of the individuals of the population                                                                *
************************************************************************************************************************/
void Clean_population()
{

    long int i, j;

    for (i = 0; i < P_size; i++)
    {
        for (j = 0; j < number_items + 5; j++)
        {
            population[i][j].L.free_linked_list();
            population[i][j].Bin_Fullness = 0;
            children[i][j].L.free_linked_list();
            children[i][j].Bin_Fullness = 0;
        }
    }
}

/************************************************************************************************************************
 To check if any of the items of the current bin is already in the solution                                             *
 Input:                                                                                                                 *
    The position in the population of the solution: individual                                                          *
    A new bin that could be added to the solution: bin                                                                  *
   An array that indicates the items that are already in the solution: items                                            *
 Output:                                                                                                                *
    (1) when none of the items in the current bin is already in the solution                                            *
   (0) otherwise                                                                                                        *
************************************************************************************************************************/
long int Used_Items(long int individual, long int bin, long int items[])
{

    long int item, i, counter = 0;
    node *p;

    p = population[individual][bin].L.first;
    while (p != NULL)
    {
        item = p->data;
        p = p->next;
        if (items[item] != 1)
        {
            items_auxiliary[counter++] = item;
            items[item] = 1;
        }
        else
        {
            for (i = 0; i < counter; i++)
                items[items_auxiliary[i]] = 0;
            return 0;
        }
    }
    return (1);
}

/************************************************************************************************************************
 To put together all the used bins of the solution                                                                      *
 Input:                                                                                                                 *
    The position in the population of the solution: individual                                                          *
************************************************************************************************************************/
void Adjust_Solution(long int individual)
{

    long int i = 0, j = 0, k;

    while (population[individual][i].Bin_Fullness > 0)
        i++;
    for (j = i, k = i; j < number_items; j++, k++)
    {
        if (j < population[individual][number_items + 1].Bin_Fullness)
        {
            while (population[individual][k].Bin_Fullness == 0)
                k++;
            population[individual][j].L.first = NULL;
            population[individual][j].L.last = NULL;
            population[individual][j].Bin_Fullness = population[individual][k].Bin_Fullness;
            population[individual][j].L.get_linked_list(population[individual][k].L);
        }
        else
        {
            population[individual][j].Bin_Fullness = 0;
            population[individual][j].L.first = NULL;
            population[individual][j].L.last = NULL;
            population[individual][j].L.num = 0;
        }
    }
}

/************************************************************************************************************************
 To read the data defining a BPP instance                                                                               *
************************************************************************************************************************/
long int LoadData()
{

    char string[300];
    long k;
    long int ban = 0;

    double bin_capacity1;
    double total_accumulated_aux = 0;
    FILE *data_file;

    string[0] = '\0';
    strcpy(string, file);
    if ((data_file = fopen(string, "rt")) == NULL)
    {
        printf("\nThere is no data file ==> [%s]%c", string, 7);
        return 0;
    }
    printf("\nThe file is %s\n", string);
    fgets(string, 300, data_file);
    fscanf(data_file, "%ld\n", &number_items);

    bin_capacity = 0;
    fgets(string, 300, data_file);
    fscanf(data_file, "%lf\n", &bin_capacity1);

    best_solution = 0;
    fgets(string, 300, data_file);
    fscanf(data_file, "%ld\n", &best_solution);

#ifdef VERBOSE
    printf("\n number_items : %ld \n", number_items);
    printf(" bin_capacity1 : %d \n", (int)bin_capacity1);
    printf("\n best_solution : %ld \n", best_solution);
#endif
    cout << number_items << "----" << bin_capacity1 << "----" << best_solution << endl;

    fgets(string, 300, data_file);
    total_accumulated_weight = 0;
    for (k = 0; k < number_items; k++)
    {
        fscanf(data_file, "%lf", &weight1[k]);
        weight[k] = (long int)weight1[k];
        total_accumulated_weight = (total_accumulated_weight + weight[k]);
        total_accumulated_aux += weight1[k];
#ifdef VERBOSE
        cout << " weight1[" << k << "]: " << weight1[k] << endl;
        cout << " weight[" << k << "]: " << weight[k] << endl;
        cout << " total_accumulated_weight :" << total_accumulated_weight << endl;
        cout << " total_accumulated_aux :" << total_accumulated_aux << endl;
#endif
        if (ban == 0)
        {
            if (weight1[k] / weight[k] > 1)
            {
                ban = 1;
            }
        }
    }

    if (ban)
    {
        total_accumulated_weight = 0;
        for (k = 0; k < number_items; k++)
        {
            weight[k] = (long int)(ceil(weight1[k] * bin_capacity1 - .5));
            total_accumulated_weight = (total_accumulated_weight + weight[k]);
        }
        bin_capacity1 *= bin_capacity1;
    }
    bin_capacity = (long int)bin_capacity1;

    cout << " bin_capacity :" << bin_capacity << endl;

    fclose(data_file);
    if (ban)
    {
        if ((long int)total_accumulated_weight != (long int)(ceil(total_accumulated_aux * sqrt(bin_capacity) - .5)))
        {
            printf("\t Error loading weights");
            exit(1);
        }
    }

    return 1;
}

/************************************************************************************************************************
 To print the performance of the procedure on a BPP instance in a data file                                             *
************************************************************************************************************************/
void WriteOutput()
{
    output = fopen(nameC, "a");
#ifdef MPI_ON
    fprintf(output, "[PROC: %d]\n%s \t %d \t %d \t %d \t %f", myrank, file, (int)L2, (int)global_best_solution[number_items + 1].Bin_Fullness, generation, TotalTime);
#else
    fprintf(output, "\n%s \t %d \t %d \t %d \t %f", file, (int)L2, (int)global_best_solution[number_items + 1].Bin_Fullness, generation, TotalTime);
#endif
    if (save_bestSolution == 1)
        sendtofile(global_best_solution);
    fclose(output);
}

/************************************************************************************************************************
 To print the global best solution in a data file                                                                       *
************************************************************************************************************************/
void sendtofile(SOLUTION best[])
{
    char string1[300],
        fil[300],
        aux[100];

    long double accumulated = 0;
    long int bin,
        ban = 1,
        item = 0,
        position = 0,
        bins[ATTRIBUTES] = {0},
        n_bins = (long int)best[number_items + 1].Bin_Fullness;

    int binError = -1,
        banError = 0;
    long int j;
    FILE *output;
    node *p;

    for (j = 0; j < n_bins; j++)
    {
        p = best[j].L.first;
        for (long item = p->data; p != NULL; p = p->next)
        {
            item = p->data;
            bins[j] += weight[item];
        }
    }

    strcpy(fil, "Details_GGA-CGT/GGA-CGT_S_(");
    strcpy(string1, file);
    // itoa(conf, aux, 10);
    snprintf(aux, sizeof(aux), "%ld", conf);
    strcat(fil, aux);
    strcat(fil, ")_");
    strcat(fil, string1);
    if ((output = fopen(fil, "w+")) == NULL)
    {
        // printf("\nThere is no data file ==> [%s]%c", file, 7);
        getchar();
        exit(1);
    }
    fprintf(output, "Instance:\t%s\n", file);
    fprintf(output, "Number of items:\t%ld\n", number_items);
    fprintf(output, "Bin capacity:\t%ld\n", bin_capacity);
    fprintf(output, "L2:\t%ld\n", L2);
    fprintf(output, "\n****************************GGA-CGT global best solution******************************\n");
    fprintf(output, "Number of bins:\n%ld\n", n_bins);
    fprintf(output, "Fitness:\n%f\n", best[number_items].Bin_Fullness);
    fprintf(output, "Optimal order of the weights:\n");

#ifdef MPI_ON
    for (bin = 0; bin < n_bins; bin++)
    {
        bins[bin] = 0;
        p = best[bin].L.first;
        while (true)
        {
            if (p == NULL)
                break;
            item = p->data;
            p = p->next;
            bins[bin] += weight[item];
            accumulated += weight[item];
            fprintf(output, "%ld\n", weight[item]);
            if (bins[bin] > bin_capacity)
            {
                printf("[%d]XXERROR the capacity of bin %ld was exceeded\n", myrank, bin);
                for (p = best[bin].L.first; p != NULL; p = p->next)
                {
                    printf("Elements in the bin is %ld[%d]: %ld\n", p->data, myrank, weight[p->data]);
                }
                binError = bin;
                getchar();
                banError = 1;
            }
        }
    }
    if (accumulated != total_accumulated_weight)
    {
        printf("ERROR inconsistent sum of weights");
        getchar();
    }
#endif
    fprintf(output, "\nDetailed solution:");
    for (j = 0; j < n_bins; j++)
    {
        if ((long int)bins[j] > (long int)bin_capacity)
            fprintf(output, " \n ********************ERROR the capacity of the bin was exceeded******************");
        fprintf(output, "\n\nBIN %ld\nFullness: %ld Gap: %ld\nStored items:\t ", j + 1, bins[j], bin_capacity - bins[j]);
        p = best[j].L.first;
        for (position = 0;; position++)
        {
            if (p == NULL)
                break;
            item = p->data;
            p = p->next;
            fprintf(output, "[Item: %ld, Weight: %ld]\t", item + 1, weight[item]);
        }
    }
    fclose(output);

    if (banError)
        exit(1);
}

/******************************************************
 Author: Adriana Cesario de Faria Alvim               *
         (alvim@inf.puc-rio.br, adriana@pep.ufrj.br ) *
*******************************************************/
/***************************************************************************
 Portable pseudo-random number generator                                   *
 Machine independent as long as the machine can represent all the integers *
         in the interval [- 2**31 + 1, 2**31 - 1].                         *
                                                                           *
 Reference: L. Schrage, "A more Portable Fortran Random Number Generator", *
            ACM Transactions on Mathematical Software, Vol. 2, No. 2,      *
            (June, 1979).                                                  *
                                                                           *
 The generator produces a sequence of positive integers, "ix",             *
      by the recursion: ix(i+1) = A * ix(i) mod P, where "P" is Mersenne   *
      prime number (2**31)-1 = 2147483647 and "A" = 7**5 = 16807. Thus all *
      integers "ix" produced will satisfy ( 0 < ix < 2147483647 ).         *
                                                                           *
 The generator is full cycle, every integer from 1 to (2**31)-2 =          *
      2147483646 is generated exactly once in the cycle.                   *
                                                                           *
 Input: integer "ix", ( 0 < ix < 2147483647 )                              *
 Ouput: real "xrand", ( 0 < xrand < 1 )                                    *
***************************************************************************/
float randp(int *ix)
{

    int xhi, xalo, leftlo, fhi, k;

    const int A = 16807;      /* = 7**5                      */
    const int P = 2147483647; /* = Mersenne prime (2**31)-1  */
    const int b15 = 32768;    /* = 2**15                     */
    const int b16 = 65536;    /* = 2**16                     */

    /* get 15 hi order bits of ix */
    xhi = *ix / b16;

    /* get 16 lo bits of ix and form lo product */
    xalo = (*ix - xhi * b16) * A;

    /* get 15 hi order bits of lo product   */
    leftlo = xalo / b16;

    /* from the 31 highest bits of full product */
    fhi = xhi * A + leftlo;

    /* get overflo past 31st bit of full product */
    k = fhi / b15;

    /* assemble all the parts and presubtract P */
    /* the parentheses are essential            */
    *ix = (((xalo - leftlo * b16) - P) + (fhi - k * b15) * b16) + k;

    /* add P back in if necessary  */
    if (*ix < 0)
        *ix = *ix + P;

    /* multiply by 1/(2**31-1) */
    return (float)(*ix * 4.656612875e-10);
}

/***********************************************************
 To generate a random integer "k" in [i,j].                *
 Input: integers "seed", "i" and "j" in [1,2147483646]     *
 Ouput: integer in [i,j]                                   *
************************************************************/
int get_rand_ij(int *seed, int i, int j)
{

    randp(seed);

    return (int)((double)*seed / ((double)2147483647 / ((double)(j - i + 1)))) + i;
}

/**************************************************************
 To generate a random integer "k" in [1,size].                *
 Input: integers "seed" and "size" in [1,2147483646]          *
 Ouput: integer in [1,size]                                   *
                                                              *
**************************************************************/
int get_rand(int *seed, int size)
{
    randp(seed);
    return (int)((double)*seed / ((double)2147483647 / ((double)size))) + 1;
}

/*************************************************
 To check the correctness of the implementation. *
 Output: correct (1) or wrong (0)                 *
**************************************************/
int trand()
{
    int i, seed;

    seed = 1;

    for (i = 0; i < 1000; i++)
        randp(&seed);

    if (seed == 522329230)
        return 1;
    else
        return 0;
}

/*************************************************
 * MAIN  PART                                    *
 *************************************************/
int main(int argc, char *argv[])
{
    char aux[10], nombreC[30], string[150];
    char *line, *token;

    int seed;
    int xxxi;

    char scheduleString[30];

#ifdef MPI_ON

    int flagToContinue = 1;

    int exitStatus;
    int *recieveExitStatus;

    double *receiveBufferDBL;  //   --> number_items + 4 * numberOfItemsToSend X MaxCommsPerSchedule
    double *sendBufferDBL;     //   --> number_items + 4
    long int *LLSendBuffer;    //    --> number_items + 4 * numberOfItemsToSend--> numberOfItemsToSend * ?????
    long int *LLReceiveBuffer; //    --> number_items + 4 * numberOfItemsToSend X MaxCommsPerSchedule--> numberOfItemsToSend * ??????
    int *offsetReceiveBuffer;  //    --> number_items + 4 * numberOfItemsToSend X MaxCommsPerSchedule
    int *offsetSendBuffer;     //    --> number_items + 4 * numberOfItemsToSend
    int *junk, *junk1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    strcpy(scheduleString, "schedule.txt");
    Read_Schedule(scheduleString);
    strcpy(scheduleString, "Configurations_HPC.txt");
    readHPCSchedule(scheduleString);
    // printf("[%d]--> NOFI: %d, EL: %d, BorW: %d, FH: %d\n", myrank, numberOfItemsToSend, epochLength, BorW, fullHalt);
    initQueue(&Send_Schedules, numScheduleSteps);
    initQueue(&Receive_Schedules, numScheduleSteps);

    srand(time(NULL)); // for random allocation

    // PREPARE BUFFERS

    recvRequests = (MPI_Request *)calloc(MaxCommsPerScheduleStep, sizeof(MPI_Request));
    statuses = (MPI_Status *)calloc(MaxCommsPerScheduleStep, sizeof(MPI_Status));
    // printf("[%ld]--> numitems: %d NOFI: %d, EL: %d, BorW: %d, FH: %d\n", myrank, number_items, numberOfItemsToSend, epochLength, BorW, fullHalt);

    sendBufferDBL = (double *)calloc(((Max_items + 5) * numberOfItemsToSend), sizeof(double));
    receiveBufferDBL = (double *)calloc(((Max_items + 5) * numberOfItemsToSend * MaxCommsPerScheduleStep), sizeof(double));

    LLSendBuffer = (long int *)calloc(((Max_items)*numberOfItemsToSend), sizeof(long int));
    LLReceiveBuffer = (long int *)calloc(((Max_items)*numberOfItemsToSend * MaxCommsPerScheduleStep), sizeof(long int));

    offsetSendBuffer = (int *)calloc(((Max_items)*numberOfItemsToSend), sizeof(int));
    offsetReceiveBuffer = (int *)calloc(((Max_items)*numberOfItemsToSend * MaxCommsPerScheduleStep), sizeof(int));

    recieveExitStatus = (int *)calloc(size, sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    // printf("[%d]<<%p>><<%p>><<%p>>\n", myrank, &sendBufferDBL, &sendBufferDBL[0], &sendBufferDBL[1]);

#ifdef VERBOSE_SCHEDULE

    for (i = 0; i < numScheduleSteps; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == 0)
            printf("[%d]ITERATION %ld\n", myrank, i);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[%d] Send sizes: %d\n", myrank, getCurrentQueueItemCount(&Send_Schedules));
        int *test = getCurrentQueueItems(&Send_Schedules);
        for (j = 0; j < getCurrentQueueItemCount(&Send_Schedules); j++)
        {
            printf("[%d] will send items to: %d\n", myrank, test[j]);
        }
        queueNext(&Send_Schedules);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0)
        printf("---------------------------------------\n");
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < numScheduleSteps; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == 0)
            printf("[%d]ITERATION %ld\n", myrank, i);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[%d] Receive sizes: %d\n", myrank, getCurrentQueueItemCount(&Receive_Schedules));
        int *test = getCurrentQueueItems(&Receive_Schedules);
        for (j = 0; j < getCurrentQueueItemCount(&Receive_Schedules); j++)
        {
            printf("[%d] will receive items from: %d\n", myrank, test[j]);
        }

        queueNext(&Receive_Schedules);
    }

#endif // VERBOSE_SCHEDULE

    if (myrank == 0)
    {

        system("mkdir Solutions_GGA-CGT");
        system("mkdir Details_GGA-CGT");

        // READING EACH CONFIGURATION IN FILE "configurations.txt", CONTAINING THE PARAMETER VALUES FOR EACH EXPERIMENT
        if ((input_Configurations = fopen("Configurations.txt", "rt")) == NULL)
        {
            printf("\n INVALID FILE(configurations)\n");
            exit(1);
        }

        fgets(string, 150, input_Configurations);
        while (fgets(string, 150, input_Configurations))
        {
            line = string;

            MPI_Barrier(MPI_COMM_WORLD);

            configSignal = 1;
            MPI_Bcast(&configSignal,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);

            token = strtok(line, "\t");
            conf = (long int)atoi(token);
            MPI_Bcast(&conf,
                      1,
                      MPI_LONG_INT,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            P_size = atoi(token);
            MPI_Bcast(&P_size,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            max_gen = atoi(token);
            MPI_Bcast(&max_gen,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            p_m = atof(token);
            MPI_Bcast(&p_m,
                      1,
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            p_c = atof(token);
            MPI_Bcast(&p_c,
                      1,
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            k_ncs = atof(token);
            MPI_Bcast(&k_ncs,
                      1,
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            k_cs = atof(token);
            MPI_Bcast(&k_cs,
                      1,
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            B_size = atof(token);
            MPI_Bcast(&B_size,
                      1,
                      MPI_DOUBLE,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            life_span = atoi(token);
            MPI_Bcast(&life_span,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);
            token = strtok(line, "\t");
            seed = atoi(token);
            MPI_Bcast(&seed,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);
#ifndef SAME_SEED_ON_ALL_PROCS
            seed = seed * (myrank * 1 + 1);
#endif
            token = strtok(line, "\t");
            save_bestSolution = atoi(token);

            MPI_Bcast(&save_bestSolution,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);

            printf("\n(%d)Running With Configuration: %ld\n", myrank, conf);
            strcpy(nameC, "Solutions_GGA-CGT/GGA-CGT_(");
            snprintf(aux, sizeof(aux), "%ld", conf);
            strcat(nameC, aux);
            strcat(nameC, ".");
            snprintf(aux, sizeof(aux), "%d", myrank);
            strcat(nameC, aux);
            strcat(nameC, ").txt");
            output = fopen(nameC, "w+");
            fprintf(output, "CONF\t|P|\tmax_gen\tn_m\tn_c\tk1(non-cloned_solutions)\tk2(cloned_solutions)\t|B|\tlife_span\tseed");
            fprintf(output, "\n%ld\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d", conf, P_size, max_gen, p_m, p_c, k_ncs, k_cs, B_size, life_span, seed);
            fprintf(output, "\nInstancias \t L2 \t Bins \t Gen \t Time");
            fclose(output);

            // READING FILE "instances.txt" CONTAINING THE NAME OF BPP INSTANCES TO PROCESS
            if ((input_Instances = fopen("instances.txt", "rt")) == NULL)
            {
                printf("\n INVALID FILE");
                exit(1);
            }

            while (fgets(file, 50, input_Instances))
            {
                // fscanf(input_Instances,"%s",file);
                line = file;
                token = strtok(line, "\n");
                *(line - 2) = '\0'; // to convert the newline character to null so that newline does not mess up the input

                MPI_Barrier(MPI_COMM_WORLD);

                instanceSignal = 1;
                MPI_Bcast(&instanceSignal,
                          1,
                          MPI_INT,
                          0,
                          MPI_COMM_WORLD);

                LoadData();
                flagToContinue = 1;

                MPI_Bcast(&number_items,
                          1,
                          MPI_LONG_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&best_solution,
                          1,
                          MPI_LONG_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&weight1,
                          number_items,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&weight,
                          number_items,
                          MPI_LONG_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&total_accumulated_weight,
                          1,
                          MPI_LONG_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&bin_capacity,
                          1,
                          MPI_LONG_INT,
                          0,
                          MPI_COMM_WORLD);

                for (i = 0; i < number_items; i++)
                    ordered_weight[i] = i;
                Sort_Descending_Weights(ordered_weight, number_items);
                LowerBound();
                seed_permutation = seed;
                seed_emptybin = seed;
                for (i = 0; i < P_size; i++)
                {
                    ordered_population[i] = i;
                    random_individuals[i] = i;
                    best_individuals[i] = i;
                }
                Clean_population();
                is_optimal_solution = 0;
                generation = 0;
                for (i = 0, j = n_; j < number_items; i++)
                    permutation[i] = ordered_weight[j++];
                repeated_fitness = 0;

                MPI_Barrier(MPI_COMM_WORLD);
                // printf("SETUP DONE [%d] ftc: %d xxxi %d\n", myrank, flagToContinue, xxxi);

                // procedure GGA-CGT
                iteration = 1;
                start = clock();

                if (Generate_Initial_Population())
                {
                    flagToContinue = 0;
                    xxxi = 1;
                }
                //           if(!Generate_Initial_Population())                //Generate_Initial_Population() returns 1 if an optimal solution is found
                for (generation = 0; generation < max_gen; generation++)
                {
                    iteration++;
                    //                   printf("[%d]iteration: %d, gen: %d epoch: %d\n",myrank, iteration, generation, epochLength);

                    // INTER-PROCESS-COMMINICATION --- START
                    if (iteration % epochLength == 0)
                    {
                        //                            printf("[%d]iteration: %d, epoch: %d\n",myrank, iteration, epochLength);
#ifdef VERBOSE_SCHEDULE
                        printf("[%d]iteration: %d, epoch: %d\n", myrank, iteration, epochLength);
#endif
                        // if epoch is correct .. prepare items
                        // if epoch is correct .. send items

                        printf("[%d] gen: %dm xxxi %d(L2: %ld - Best: %f)\n", myrank, generation, xxxi, L2, global_best_solution[number_items + 1].Bin_Fullness);

                        receiveProcs = getCurrentQueueItems(&Receive_Schedules);
                        sendProcs = getCurrentQueueItems(&Send_Schedules);
                        receiveProcCount = getCurrentQueueItemCount(&Receive_Schedules);
                        sendProcCount = getCurrentQueueItemCount(&Send_Schedules);

                        // COMMUNICATION FOR BIN FULLNESS VALUES --- START
                        //                        printf("[%d]BIN FULLNESS COMM START\n", myrank);

                        // 1.1: First, issue IReceives for each of the processors that will send us something
                        // printf("[%d]>>>>>%d<<<<<<\n",myrank, getCurrentQueueItemCount(&Receive_Schedules));
                        for (j = 0; j < receiveProcCount; j++)
                        {
                            //  printf("IT: %d[%d]****************(I am %d)expecting from: %d(%p)\n",iteration, myrank, myrank, test[j], &test[j]);
                            //                           printf("[%d] IT: %d Bin Comm: will be receiving from: %d\n", myrank, iteration, receiveProcs[j]);
                            MPI_Irecv(&receiveBufferDBL[j * ((number_items + 5) * numberOfItemsToSend)],
                                      numberOfItemsToSend * (number_items + 5),
                                      MPI_DOUBLE,
                                      receiveProcs[j],
                                      iteration,
                                      MPI_COMM_WORLD,
                                      &recvRequests[j]);
                        }

                        // 1.2: Second, BinFullNess Values will be send - prepare
                        xcount = 0;
                        for (i = 0; i < numberOfItemsToSend; i++)
                        {
                            for (j = 0; j < number_items + 5; j++)
                            {
                                //                               printf("IT: %d[%d]what I send is: [%d]: %f \n", iteration, myrank, j, population[ ordered_population[P_size - 1 - i] ][j].Bin_Fullness);
                                sendBufferDBL[xcount] = population[ordered_population[P_size - 1 - i]][j].Bin_Fullness;
                                xcount++;
                            }
                        }

                        // 1.3: Second, send it to all processors waiting a message from us.
                        for (j = 0; j < sendProcCount; j++)
                        {
                            //                           printf("[%d] IT: %d Bin Comm: will be sending to: %d\n", myrank, iteration, sendProcs[j]);
                            //                           for (i = 0; i < number_items+5; i++ )
                            //                               printf("IT: %d [%d] sending item(%d): %f\n", iteration, myrank,i,  sendBufferDBL[i]);
                            MPI_Send(&sendBufferDBL[0],
                                     numberOfItemsToSend * (number_items + 5),
                                     MPI_DOUBLE,
                                     sendProcs[j],
                                     iteration,
                                     MPI_COMM_WORLD);
                        }

                        // 1.4: Third, wait for each Irecv to be received completely.
                        // test = getCurrentQueueItems(&Receive_Schedules);
                        for (i = 0; i < receiveProcCount; i++)
                        {
                            // printf("[%d] waiting for message from %d(%p)\n", myrank, test[i], &test[i]);
                            MPI_Wait(&recvRequests[i], &statuses[i]);
                        }

                        //                       for ( j = 0; j < receiveProcCount  ; j++ )
                        //                           for (i = 0; i < number_items+5; i++ )
                        //                               printf("IT: %d [%d] received item(%d) from %d: %f\n", iteration, myrank,i,  receiveBufferDBL[i], receiveProcs[j]);

                        // BIN COMMUNICATION DONE - NOW UNPACK ITEMS
                        // 1.5: compute where to write them
                        if (BorW == 0)
                        {
                            for (i = 0; i < numberOfItemsToSend; i++)
                            {
                                whereToWritePopulation[i] = P_size - 1 - i;
                            }
                        }
                        else if (BorW == 1)
                        {
                            for (i = 0; i < numberOfItemsToSend; i++)
                            {
                                whereToWritePopulation[i] = i;
                            }
                        }
                        else
                        { // BorW is == 2
                            for (i = 0; i < numberOfItemsToSend; i++)
                            {
                                whereToWritePopulation[i] = rand() % P_size;
                            }
                        }

                        // 1.6: now go and write binFulllnesses to relevant items.
#ifdef UNPACK
                        int count = 0;
                        for (k = 0; k < receiveProcCount; k++)
                        {
                            for (i = 0; i < numberOfItemsToSend; i++)
                            {
                                for (j = 0; j <= number_items + 4; j++)
                                {
                                    population[ordered_population[whereToWritePopulation[i]]][j].Bin_Fullness = receiveBufferDBL[count];
                                    //                                   printf("IT: %d [%d]what I receive is: [%ld]: %f \n", iteration, myrank, j, receiveBufferDBL[count]);
                                    count++;
                                }
                            }
                        }
#endif
                        // COMMUNICATION FOR BIN FULLNESS VALUES --- END
                        //                        printf("[%d]BIN FULLNESS COMM END\n", myrank);

                        MPI_Barrier(MPI_COMM_WORLD);

                        // COMMUNICATION FOR THE LINKED LISTS -- START

                        // OFFSETS FIRST WHILE PREPARING THE VALUES ALSO
                        // 2.1: First, issue IReceives for each of the processors that will send us something
                        for (j = 0; j < receiveProcCount; j++)
                        {
#ifdef VERBOSE_SCHEDULE
                            printf("[%d] IT: %d offset Comm: will be receiving from: %d\n", myrank, iteration, receiveProcs[j]);
#endif
                            MPI_Irecv(&offsetReceiveBuffer[j * ((number_items)*numberOfItemsToSend)],
                                      numberOfItemsToSend * (number_items),
                                      MPI_INT,
                                      receiveProcs[j],
                                      0,
                                      MPI_COMM_WORLD,
                                      &recvRequests[j]);
                        }
                        // 2.2: Second, perpare the send buffer and start to send each of the linked lists
                        totalCount = 0;
                        for (i = 0; i < numberOfItemsToSend; i++)
                        {
                            for (j = 0; j < number_items; j++)
                            {
#ifdef VERBOSE_SCHEDULE
                                printf(":IT: %d[%d] I am sending my offsets and they are: [%d]: %d\n", iteration, myrank, j, population[ordered_population[P_size - 1 - i]][j].L.num);
#endif
                                offsetSendBuffer[i * number_items + j] = population[ordered_population[P_size - 1 - i]][j].L.num;
                                temp = population[ordered_population[P_size - 1 - i]][j].L.first;
                                // if (temp != NULL) printf("<(%d)%ld,%d--> %d>\n",numberOfItemsToSend,j,  population[ ordered_population[P_size - 1 - i] ][j].L.num, count);
                                tcount = 0;
                                for (k = 0; k < population[ordered_population[P_size - 1 - i]][j].L.num; k++)
                                {
#ifdef VERBOSE_SCHEDULE
                                    printf("[%d]:IT: %d and the data for bin %d is: %ld, weight: %d\n", myrank, iteration, j, temp->data, weight[temp->data]);
#endif
                                    LLSendBuffer[totalCount + tcount] = temp->data;
                                    temp = temp->next;
                                    tcount++;
                                }
                                // if (temp == NULL) printf("<(%d)%ld,%d--> %d>\n",numberOfItemsToSend,j,  population[ ordered_population[P_size - 1 - i] ][j].L.num, count);
                                totalCount = totalCount + tcount;
                            }
                        }

                        // 2.3: Second, SEND the messages to appropriate processors
                        for (j = 0; j < sendProcCount; j++)
                        {
#ifdef VERBOSE_SCHEDULE
                            printf("[%d] IT: %d Offset Comm: will be sending to: %d\n", myrank, iteration, sendProcs[j]);
#endif
                            MPI_Send(&offsetSendBuffer[0],
                                     numberOfItemsToSend * number_items,
                                     MPI_INT,
                                     sendProcs[j],
                                     0,
                                     MPI_COMM_WORLD);
                        }

                        // 2.4: Third, wait for each Irecv to finalize
                        for (i = 0; i < receiveProcCount; i++)
                        {
                            MPI_Wait(&recvRequests[i], &statuses[i]);
                        }

                        //                       for (i =0 ; i < receiveProcCount; i++ ) {
                        //                           for (j = 0; j < numberOfItemsToSend ; j++ ) {
                        //                               for (k = 0; k < number_items ; k ++ ) {
                        //                                   printf("[%d] IT: %d: Offsets I received are: Offset[%d]: %d\n", myrank, iteration, k,
                        //                                                        offsetReceiveBuffer[(i * ((number_items)* numberOfItemsToSend )) + (j * number_items) + k]);
                        //                               }
                        //                           }
                        //                       }

                        //  for (i = 0; i < number_items ;i++ ){
                        //      printf("IT %d:[%d] I have received my offsets and they are: [%d]: %d\n", iteration, myrank, i, offsetReceiveBuffer[i]);
                        //  }

                        // NOW FOR THE VALUES COMMUNICATION

                        // !!!!!!!!!!!!!!!!
                        // SEND RECEIVE VALUES MAY BE WRONG HERE
                        // !!!!!!!!!!!!!!!!!

                        // 3.1: First, issue IReceives for each of the processors that will send us something
                        for (j = 0; j < receiveProcCount; j++)
                        {
                            MPI_Irecv(&LLReceiveBuffer[j * ((number_items)*numberOfItemsToSend)],
                                      numberOfItemsToSend * number_items,
                                      MPI_LONG_INT,
                                      receiveProcs[j],
                                      0,
                                      MPI_COMM_WORLD,
                                      &recvRequests[j]);
                        }

                        // 3.2: Second, Send messages to appropriate processors
                        for (j = 0; j < sendProcCount; j++)
                        {
                            MPI_Send(&LLSendBuffer[0],
                                     numberOfItemsToSend * number_items,
                                     MPI_LONG_INT,
                                     sendProcs[j],
                                     0,
                                     MPI_COMM_WORLD);
                        }

                        // 3.3: Third, wait for each Irecv to finalize
                        for (i = 0; i < receiveProcCount; i++)
                        {
                            MPI_Wait(&recvRequests[i], &statuses[i]);
                        }

                        // 3.4: Fourth, unpack the items
#ifdef UNPACK
                        for (l = 0; l < receiveProcCount; l++)
                        {
                            count = 0;
                            for (i = 0; i < numberOfItemsToSend; i++)
                            {
                                for (j = 0; j < number_items; j++)
                                {
                                    population[ordered_population[whereToWritePopulation[i]]][j].L.free_linked_list();
                                    for (k = 0; k < offsetReceiveBuffer[(l * number_items * numberOfItemsToSend) + (i * number_items) + j]; k++)
                                    {
                                        //                                      printf("IT: %d-->[%d]--> received and unpacking (%ld)(%d) %ld\n", iteration, myrank, j,count,
                                        //                                                                        LLReceiveBuffer[ (numberOfItemsToSend * number_items * l) + count ]);
                                        population[ordered_population[whereToWritePopulation[i]]][j].L.insert(LLReceiveBuffer[(numberOfItemsToSend * number_items * l) + count]);
                                        count++;
                                    }
                                }
                            }
                        }
#endif
                        // COMMUNICATION FOR THE LINKED LISTS -- END
                        tcount = 0;
                        for (i = 0; i < receiveProcCount; i++)
                        {
                            for (j = 0; j < numberOfItemsToSend; j++)
                            {
                                for (k = 0; k < number_items; k++)
                                {
                                    population[ordered_population[whereToWritePopulation[j]]][k].L.free_linked_list();
#ifdef VERBOSE_SCHEDULE
                                    printf("[%d] IT: %d: Offsets I received are: Offset[%d]: %d BinFullness[%d]: %f\n", myrank, iteration, k,
                                           offsetReceiveBuffer[(i * ((number_items)*numberOfItemsToSend)) + (j * number_items) + k], k,
                                           receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + k]);
#endif
                                    population[ordered_population[whereToWritePopulation[j]]][k].Bin_Fullness = receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + k];
                                    for (l = 0; l < offsetReceiveBuffer[(i * ((number_items)*numberOfItemsToSend)) + (j * number_items) + k]; l++)
                                    {
#ifdef VERBOSE_SCHEDULE
                                        printf("[%d] IT: %d: Item: %ld Item_Weight: %ld \n", myrank, iteration, LLReceiveBuffer[tcount], weight[LLReceiveBuffer[tcount]]);
#endif
                                        population[ordered_population[whereToWritePopulation[j]]][k].L.insert(LLReceiveBuffer[tcount]);
                                        tcount++;
                                    }
                                }
                                population[ordered_population[whereToWritePopulation[j]]][number_items].Bin_Fullness = receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items];
                                population[ordered_population[whereToWritePopulation[j]]][number_items + 1].Bin_Fullness = receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 1];
                                population[ordered_population[whereToWritePopulation[j]]][number_items + 2].Bin_Fullness = receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 2];
                                population[ordered_population[whereToWritePopulation[j]]][number_items + 3].Bin_Fullness = receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 3];
                                population[ordered_population[whereToWritePopulation[j]]][number_items + 4].Bin_Fullness = receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 4];

#ifdef VERBOSE_SCHEDULE
                                printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items,
                                       receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items]);
                                printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 1,
                                       receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 1]);
                                printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 2,
                                       receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 2]);
                                printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 3,
                                       receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 3]);
                                printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 4,
                                       receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 4]);
#endif
                            }
                        }

                        // Sort_Ascending_IndividualsFitness();

                        // Move to next schedule step
                        queueNext(&Receive_Schedules);
                        queueNext(&Send_Schedules);

                        MPI_Barrier(MPI_COMM_WORLD);
                        // printf("\n ------------------------ \n\n");
                        //                       printf("[%d]INTERPROCESS COMM END\n", myrank);

                        MPI_Barrier(MPI_COMM_WORLD);

                    } // if epoclength % = 0
                      // INTER-PROCESS-COMMINICATION --- END

                    if (flagToContinue)
                    {
                        xxxi = Generation(); // Generation() returns 1 if an optimal solution was found
                        //                       printf("[%d](Gen: %d)flagToContinue : %d, xxxi: %d\n", myrank, generation, flagToContinue, xxxi);
                        if (best_solution == global_best_solution[number_items + 1].Bin_Fullness)
                        {
                            Find_Best_Solution();
                            printf("BSC[%d] gen: %dm xxxi %d(L2: %ld - Best: %f, real best: %ld)\n", myrank, generation, xxxi, L2, global_best_solution[number_items + 1].Bin_Fullness, best_solution);
                            xxxi = 1;
                        }

                        if (xxxi == 1)
                        {
                            flagToContinue = 0;
                        }
                        else if (xxxi == 2)
                        {
                            Find_Best_Solution();
                            //                              printf("[0] gen: %dm xxxi %d(L2: %ld - Best: %f)\n", generation, xxxi, L2, global_best_solution[number_items+1].Bin_Fullness);
                            flagToContinue = 0;
                            if (best_solution == global_best_solution[number_items + 1].Bin_Fullness)
                                xxxi = 1;
                            else
                                xxxi = 2;
                        } // hakan's edit
                    }
                    else
                    {
                        //                      printf("[%d](Gen: %d)flag is not continuing optimal is: %d\n", myrank, generation, is_optimal_solution);
                    }
                    // printf("GATHERING! [%d]\n", myrank);

                    MPI_Allgather(&xxxi,
                                  1,
                                  MPI_INT,
                                  recieveExitStatus,
                                  1,
                                  MPI_INT,
                                  MPI_COMM_WORLD);
                    // printf("GATHERED! [%d]\n", myrank);

                    for (i = 0; i < size; i++)
                    {
                        if (recieveExitStatus[i] != 0)
                            xxxi = recieveExitStatus[i];
                    }

                    if (xxxi)
                    {
                        xxxi = 0;
                        break;
                    }

#ifdef VERBOSE
                    cout << "Generation : " << generation << endl;
#else
                    if (generation % 100 == 0)
                        cout << "Generation : " << generation << endl;
#endif
                    Find_Best_Solution();
                }

                if (!is_optimal_solution)
                { // is_optimal_solution is 1 if an optimal solution was printed before
                    clock_t end = clock();
                    TotalTime = (end - start) / (1000 * 1.0);
                    Find_Best_Solution();
                    printf("[%d]Optimal Found at iteration %d\n", myrank, generation);
                    WriteOutput();
                }
                else
                    printf("X[%d]Optimal Found at iteration %d\n", myrank, generation);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            instanceSignal = 0;
            MPI_Bcast(&instanceSignal,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);
            fclose(input_Instances);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        configSignal = 0;
        MPI_Bcast(&configSignal,
                  1,
                  MPI_INT,
                  0,
                  MPI_COMM_WORLD);
        fclose(input_Configurations);

#ifdef PRINT_POPULATION_CONTENTS
        FILE *popOutFile;

        popOutFile = fopen("population.txt", "w");
        if (!popOutFile)
        {
            cout << "Population file cannot be opened" << endl;
            exit(0);
        }

        printAllPopulation(*population, popOutFile, n_bins);
        fclose(popOutFile);

#endif
    }
    else
    { // if myrank != 0
        configSignal = 1;
        while (configSignal)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&configSignal,
                      1,
                      MPI_INT,
                      0,
                      MPI_COMM_WORLD);
            if (configSignal)
            {
                MPI_Bcast(&conf,
                          1,
                          MPI_LONG_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&P_size,
                          1,
                          MPI_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&max_gen,
                          1,
                          MPI_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&p_m,
                          1,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&p_c,
                          1,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&k_ncs,
                          1,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&k_cs,
                          1,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&B_size,
                          1,
                          MPI_DOUBLE,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&life_span,
                          1,
                          MPI_INT,
                          0,
                          MPI_COMM_WORLD);
                MPI_Bcast(&seed,
                          1,
                          MPI_INT,
                          0,
                          MPI_COMM_WORLD);
#ifndef SAME_SEED_ON_ALL_PROCS
                seed = seed * (myrank * 2 + 1);
#endif
                MPI_Bcast(&save_bestSolution,
                          1,
                          MPI_INT,
                          0,
                          MPI_COMM_WORLD);

                printf("\n(%d)RunningF With Configuration: %ld\n", myrank, conf);
                strcpy(nameC, "Solutions_GGA-CGT/GGA-CGT_(");
                snprintf(aux, sizeof(aux), "%ld", conf);
                strcat(nameC, aux);
                strcat(nameC, ".");
                snprintf(aux, sizeof(aux), "%d", myrank);
                strcat(nameC, aux);
                strcat(nameC, ").txt");
                output = fopen(nameC, "w+");
                fprintf(output, "CONF\t|P|\tmax_gen\tn_m\tn_c\tk1(non-cloned_solutions)\tk2(cloned_solutions)\t|B|\tlife_span\tseed");
                fprintf(output, "\n%ld\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d", conf, P_size, max_gen, p_m, p_c, k_ncs, k_cs, B_size, life_span, seed);
                fprintf(output, "\nInstancias \t L2 \t Bins \t Gen \t Time");
                fclose(output);

                instanceSignal = 1;
                while (instanceSignal)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Bcast(&instanceSignal,
                              1,
                              MPI_INT,
                              0,
                              MPI_COMM_WORLD);
                    if (instanceSignal)
                    {
                        flagToContinue = 1;
                        MPI_Bcast(&number_items,
                                  1,
                                  MPI_LONG_INT,
                                  0,
                                  MPI_COMM_WORLD);
                        MPI_Bcast(&best_solution,
                                  1,
                                  MPI_LONG_INT,
                                  0,
                                  MPI_COMM_WORLD);
                        MPI_Bcast(&weight1,
                                  number_items,
                                  MPI_DOUBLE,
                                  0,
                                  MPI_COMM_WORLD);
                        MPI_Bcast(&weight,
                                  number_items,
                                  MPI_LONG_INT,
                                  0,
                                  MPI_COMM_WORLD);
                        MPI_Bcast(&total_accumulated_weight,
                                  1,
                                  MPI_LONG_INT,
                                  0,
                                  MPI_COMM_WORLD);
                        MPI_Bcast(&bin_capacity,
                                  1,
                                  MPI_LONG_INT,
                                  0,
                                  MPI_COMM_WORLD);

                        for (i = 0; i < number_items; i++)
                            ordered_weight[i] = i;
                        Sort_Descending_Weights(ordered_weight, number_items);
                        LowerBound();
                        seed_permutation = seed;
                        seed_emptybin = seed;
                        for (i = 0; i < P_size; i++)
                        {
                            ordered_population[i] = i;
                            random_individuals[i] = i;
                            best_individuals[i] = i;
                        }
                        Clean_population();

                        is_optimal_solution = 0;
                        generation = 0;
                        for (i = 0, j = n_; j < number_items; i++)
                            permutation[i] = ordered_weight[j++];
                        repeated_fitness = 0;

                        MPI_Barrier(MPI_COMM_WORLD);
                        // printf("SETUP DONE [%d] ftc: %d xxxi %d\n", myrank, flagToContinue, xxxi);

                        // procedure GGA-CGT
                        iteration = 1;
                        start = clock();

                        if (Generate_Initial_Population())
                        {
                            flagToContinue = 0;
                            xxxi = 1;
                        }
                        //                    if(!Generate_Initial_Population())                //Generate_Initial_Population() returns 1 if an optimal solution is found
                        for (generation = 0; generation < max_gen; generation++)
                        {
                            iteration++;
                            //                                           printf("[%d]iteration: %d, gen: %d epoch: %d\n",myrank, iteration, generation, epochLength);

                            // INTER-PROCESS-COMMINICATION --- START
                            if (iteration % epochLength == 0)
                            {

                                printf("[%d] gen: %dm xxxi %d(L2: %ld - Best: %f)\n", myrank, generation, xxxi, L2, global_best_solution[number_items + 1].Bin_Fullness);

                                receiveProcs = getCurrentQueueItems(&Receive_Schedules);
                                sendProcs = getCurrentQueueItems(&Send_Schedules);
                                receiveProcCount = getCurrentQueueItemCount(&Receive_Schedules);
                                sendProcCount = getCurrentQueueItemCount(&Send_Schedules);

                                // if epoch is correct .. prepare items
                                // if epoch is correct .. send items

                                // COMMUNICATION FOR BIN FULLNESS VALUES --- START

                                // 1.1: First, issue IReceives for each of the processors that will send us something
                                // printf("[%d]>>>>>%d<<<<<<\n",myrank, getCurrentQueueItemCount(&Receive_Schedules));
                                for (j = 0; j < receiveProcCount; j++)
                                {
                                    // printf("IT: %d *****************(I am %d)expecting from: %d(%p)\n", iteration, myrank,  test[j], &test[j]);
                                    //                               printf("[%d] IT: %d Bin Comm: will be receiving from: %d\n", myrank, iteration, receiveProcs[j]);
                                    MPI_Irecv(&receiveBufferDBL[j * ((number_items + 5) * numberOfItemsToSend)],
                                              numberOfItemsToSend * (number_items + 5),
                                              MPI_DOUBLE,
                                              receiveProcs[j],
                                              iteration,
                                              MPI_COMM_WORLD,
                                              &recvRequests[j]);
                                }

                                // 1.2: Second, BinFullNess Values will be send - prepare
                                xcount = 0;
                                for (i = 0; i < numberOfItemsToSend; i++)
                                {
                                    for (j = 0; j <= number_items + 4; j++)
                                    {
                                        //       printf("IT: %d[%d]what I send is: [%d]: %f \n", iteration, myrank, j, population[ ordered_population[P_size - 1 - i] ][j].Bin_Fullness);
                                        sendBufferDBL[xcount] = population[ordered_population[P_size - 1 - i]][j].Bin_Fullness;
                                        xcount++;
                                    }
                                }
                                // printf("NI: %ld Number of Itens to Send: %d --> count: %d\n", number_items, numberOfItemsToSend, count);

                                // 1.3: Second, send it to all processors waiting a message from us.
                                for (j = 0; j < sendProcCount; j++)
                                {
                                    //                               printf("[%d] IT: %d Bin Comm: will be sending to: %d\n", myrank, iteration, sendProcs[j]);
                                    //                               for (i = 0; i < number_items+5; i++ )
                                    //                                   printf("IT: %d [%d] sending item(%d): %f\n", iteration, myrank,i,  sendBufferDBL[i]);
                                    MPI_Send(&sendBufferDBL[0],
                                             numberOfItemsToSend * (number_items + 5),
                                             MPI_DOUBLE,
                                             sendProcs[j],
                                             iteration,
                                             MPI_COMM_WORLD);
                                }

                                // 1.4: Third, wait for each Irecv to be received completely.
                                // test = getCurrentQueueItems(&Receive_Schedules);
                                for (i = 0; i < receiveProcCount; i++)
                                {
                                    //    printf("[%d] waiting for message from %d(%p)\n", myrank, test[i], &test[i]);
                                    MPI_Wait(&recvRequests[i], &statuses[i]);
                                }
                                // printf("proc %d finished\n", myrank);

                                //                      for ( j = 0; j < receiveProcCount  ; j++ )
                                //                          for (i = 0; i < number_items+5; i++ )
                                //                              printf("IT: %d[%d] received item(%d) from %d: %f\n", iteration, myrank,i,  receiveBufferDBL[i], receiveProcs[j]);

                                // BIN COMMUNICATION DONE - NOW UNPACK ITEMS

                                // 1.5: compute where to write them
                                if (BorW == 0)
                                {
                                    for (i = 0; i < numberOfItemsToSend; i++)
                                    {
                                        whereToWritePopulation[i] = P_size - 1 - i;
                                    }
                                }
                                else if (BorW == 1)
                                {
                                    for (i = 0; i < numberOfItemsToSend; i++)
                                    {
                                        whereToWritePopulation[i] = i;
                                    }
                                }
                                else
                                { // BorW is == 2
                                    for (i = 0; i < numberOfItemsToSend; i++)
                                    {
                                        whereToWritePopulation[i] = rand() % P_size;
                                    }
                                }

                                // 1.6: now go and write binFulllnesses to relevant items.
#ifdef UNPACK
                                count = 0;
                                for (k = 0; k < receiveProcCount; k++)
                                {
                                    for (i = 0; i < numberOfItemsToSend; i++)
                                    {
                                        for (j = 0; j < number_items + 5; j++)
                                        {
                                            //                                       printf("IT: %d [%d]what I receive is: [%ld]: %f \n", iteration, myrank, j, receiveBufferDBL[count]);
                                            population[ordered_population[whereToWritePopulation[i]]][j].Bin_Fullness = receiveBufferDBL[count];
                                            count++;
                                        }
                                    }
                                }
#endif
                                // COMMUNICATION FOR BIN FULLNESS VALUES --- END

                                MPI_Barrier(MPI_COMM_WORLD);

                                // COMMUNICATION FOR THE LINKED LISTS -- START

                                // OFFSETS FIRST WHILE PREPARING THE VALUES ALSO
                                // 2.1: First, issue IReceives for each of the processors that will send us something
                                for (j = 0; j < receiveProcCount; j++)
                                {
                                    //                               printf("[%d] IT: %d Offset Comm: will be receiving from: %d\n", myrank, iteration, receiveProcs[j]);
                                    MPI_Irecv(&offsetReceiveBuffer[j * ((number_items)*numberOfItemsToSend)],
                                              numberOfItemsToSend * (number_items),
                                              MPI_INT,
                                              receiveProcs[j],
                                              0,
                                              MPI_COMM_WORLD,
                                              &recvRequests[j]);
                                }

                                // 2.2: Second, perpare the send buffer and start to send each of the linked lists
                                totalCount = 0;
                                for (i = 0; i < numberOfItemsToSend; i++)
                                {
                                    for (j = 0; j < (number_items); j++)
                                    {
                                        //                                   printf(":IT: %d [%d] I am sending my offsets and they are: [%d]: %d\n", iteration, myrank, j, population[ ordered_population[P_size - 1 - i] ][j].L.num$
                                        offsetSendBuffer[i * number_items + j] = population[ordered_population[P_size - 1 - i]][j].L.num;
                                        temp = population[ordered_population[P_size - 1 - i]][j].L.first;
                                        int count = 0;
                                        for (k = 0; k < population[ordered_population[P_size - 1 - i]][j].L.num; k++)
                                        {
                                            //                                      printf("[%d] :IT: %d and the data for bin %d is: %ld, weight: %d\n",  myrank, iteration,j, temp->data, weight[temp->data]);
                                            LLSendBuffer[totalCount + count] = temp->data;
                                            temp = temp->next;
                                            count++;
                                        }
                                        totalCount = totalCount + count;
                                    }
                                }

                                // 2.3: Second, SEND the messages to appropriate processors
                                for (j = 0; j < sendProcCount; j++)
                                {
                                    //                               printf("[%d] IT: %d Offset Comm: will be sending to: %d\n", myrank, iteration, sendProcs[j]);
                                    MPI_Send(&offsetSendBuffer[0],
                                             numberOfItemsToSend * number_items,
                                             MPI_INT,
                                             sendProcs[j],
                                             0,
                                             MPI_COMM_WORLD);
                                }

                                // 2.4: Third, wait for each Irecv to finalize
                                for (i = 0; i < receiveProcCount; i++)
                                {
                                    MPI_Wait(&recvRequests[i], &statuses[i]);
                                }

                                //                           for (i =0 ; i < receiveProcCount; i++ ) {
                                //                               for (j = 0; j < numberOfItemsToSend ; j++ ) {
                                //                                   for (k = 0; k < number_items ; k ++ ) {
                                //                                       printf("[%d] IT: %d: Offsets I received are: Offset[%d]: %d\n", myrank, iteration, k,
                                //                                                                                                     offsetReceiveBuffer[(i * ((number_items)* numberOfItemsToSend )) + (j * number_items) +$
                                //                                   }
                                //                               }
                                //                           }

                                //                           for (i = 0; i < number_items ;i++ ){
                                //                               printf(": IT %d:[%d] I have received my offsets and they are: [%d]: %d\n", iteration, myrank, i, offsetReceiveBuffer[i]);
                                //                           }
                                // NOW FOR THE VALUES COMMUNICATION

                                // !!!!!!!!!!!!!!!!
                                // SEND RECEIVE VALUES MAY BE WRONG HERE
                                // !!!!!!!!!!!!!!!!!

                                // 3.1: First, issue IReceives for each of the processors that will send us something
                                // test = getCurrentQueueItems(&Receive_Schedules);
                                for (j = 0; j < receiveProcCount; j++)
                                {
                                    MPI_Irecv(&LLReceiveBuffer[j * ((number_items)*numberOfItemsToSend)],
                                              numberOfItemsToSend * number_items,
                                              MPI_LONG_INT,
                                              receiveProcs[j],
                                              0,
                                              MPI_COMM_WORLD,
                                              &recvRequests[j]);
                                }

                                // 3.2: test = getCurrentQueueItems(&Send_Schedules);
                                for (j = 0; j < sendProcCount; j++)
                                {
                                    MPI_Send(&LLSendBuffer[0],
                                             numberOfItemsToSend * number_items,
                                             MPI_LONG_INT,
                                             sendProcs[j],
                                             0,
                                             MPI_COMM_WORLD);
                                }

                                // 3.3: Third, wait for each Irecv to finalize
                                for (i = 0; i < receiveProcCount; i++)
                                {
                                    MPI_Wait(&recvRequests[i], &statuses[i]);
                                }

                                // 3.4: Fourth, unpack the items
#ifdef UNPACK
                                for (l = 0; l < receiveProcCount; l++)
                                {
                                    count = 0;
                                    for (i = 0; i < numberOfItemsToSend; i++)
                                    {
                                        for (j = 0; j < (number_items); j++)
                                        {
                                            population[ordered_population[whereToWritePopulation[i]]][j].L.free_linked_list();
                                            for (k = 0; k < offsetReceiveBuffer[(l * number_items * numberOfItemsToSend) + (i * number_items) + j]; k++)
                                            {
                                                //                                                                                    printf("IT: %d-->[%d]--> received and unpacking (%ld)(%d) %ld\n", iteration, myrank, j, count,
                                                //                                                                                                              LLReceiveBuffer[(numberOfItemsToSend * number_items * l) + count ]);
                                                population[ordered_population[whereToWritePopulation[i]]][j].L.insert(LLReceiveBuffer[(numberOfItemsToSend * number_items * l) + count]);
                                                count++;
                                            }
                                        }
                                    }
                                }
#endif

                                //                           // COMMUNICATION FOR THE LINKED LISTS -- END

                                count = 0;
                                for (i = 0; i < receiveProcCount; i++)
                                {
                                    for (j = 0; j < numberOfItemsToSend; j++)
                                    {
                                        for (k = 0; k < number_items; k++)
                                        {
                                            population[ordered_population[whereToWritePopulation[j]]][k].L.free_linked_list();
                                            //                                  printf("[%d] IT: %d: Offsets I received are: Offset[%d]: %d BinFullness[%d]: %f\n", myrank, iteration, k,
                                            //                                                                                                  offsetReceiveBuffer[(i * ((number_items)* numberOfItemsToSend )) +
                                            //                                                                                                    (j * number_items) + k], k,
                                            //                                                                                                  receiveBufferDBL[(i * ((number_items + 5)* numberOfItemsToSend )) +
                                            //                                                                                                     (j * (number_items + 5)) + k]);
                                            population[ordered_population[whereToWritePopulation[j]]][k].Bin_Fullness =
                                                receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + k];
                                            for (l = 0; l < offsetReceiveBuffer[(i * ((number_items)*numberOfItemsToSend)) + (j * number_items) + k]; l++)
                                            {
                                                //                                      printf("[%d] IT: %d: Item: %ld Item_Weight: %ld \n", myrank, iteration, LLReceiveBuffer[count], weight[LLReceiveBuffer[count]]);
                                                population[ordered_population[whereToWritePopulation[j]]][k].L.insert(LLReceiveBuffer[count]);
                                                count++;
                                            }
                                        }
                                        population[ordered_population[whereToWritePopulation[j]]][number_items].Bin_Fullness =
                                            receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items];
                                        population[ordered_population[whereToWritePopulation[j]]][number_items + 1].Bin_Fullness =
                                            receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 1];
                                        population[ordered_population[whereToWritePopulation[j]]][number_items + 2].Bin_Fullness =
                                            receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 2];
                                        population[ordered_population[whereToWritePopulation[j]]][number_items + 3].Bin_Fullness =
                                            receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 3];
                                        population[ordered_population[whereToWritePopulation[j]]][number_items + 4].Bin_Fullness =
                                            receiveBufferDBL[(i * ((number_items + 5) * numberOfItemsToSend)) + (j * (number_items + 5)) + number_items + 4];

                                        //                               printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items,
                                        //                                                                         receiveBufferDBL[(i * ((number_items + 5)* numberOfItemsToSend )) + (j * (number_items + 5)) + number_items]);
                                        //                               printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 1,
                                        //                                                                         receiveBufferDBL[(i * ((number_items + 5)* numberOfItemsToSend )) + (j * (number_items + 5)) + number_items + 1]);
                                        //                               printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 2,
                                        //                                                                         receiveBufferDBL[(i * ((number_items + 5)* numberOfItemsToSend )) + (j * (number_items + 5)) + number_items + 2]);
                                        //                               printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 3,
                                        //                                                                         receiveBufferDBL[(i * ((number_items + 5)* numberOfItemsToSend )) + (j * (number_items + 5)) + number_items + 3]);
                                        //                               printf("[%d] IT: %d Binfullness[%d]: %f\n", myrank, iteration, number_items + 4,
                                        //                                                                         receiveBufferDBL[(i * ((number_items + 5)* numberOfItemsToSend )) + (j * (number_items + 5)) + number_items + 4]);
                                    }
                                }

                                // Sort_Ascending_IndividualsFitness();

                                // Move to next schedule step
                                queueNext(&Receive_Schedules);
                                queueNext(&Send_Schedules);
                                MPI_Barrier(MPI_COMM_WORLD);
                                MPI_Barrier(MPI_COMM_WORLD);
                            } // if epochlength % = 0
                              //  INTER-PROCESS-COMMINICATION --- END

                            if (flagToContinue)
                            {
                                xxxi = Generation(); // Generation() returns 1 if an optimal solution was found
                                if (best_solution == global_best_solution[number_items + 1].Bin_Fullness)
                                {
                                    Find_Best_Solution();
                                    printf("BSC[%d] gen: %dm xxxi %d(L2: %ld - Best: %f, real best: %ld)\n", myrank, generation, xxxi, L2, global_best_solution[number_items + 1].Bin_Fullness, best_solution);
                                    xxxi = 1;
                                }

                                if (xxxi == 1)
                                {
                                    flagToContinue = 0;
                                }
                                // else if(xxxi == 2) {flagToContinue = 0; xxxi = 0;}
                                else if (xxxi == 2)
                                {
                                    Find_Best_Solution();
                                    printf("[%d] gen: %dm xxxi %d(L2: %ld - Best: %f)\n", myrank, generation, xxxi, L2, global_best_solution[number_items + 1].Bin_Fullness);
                                    flagToContinue = 0;
                                    if (best_solution == global_best_solution[number_items + 1].Bin_Fullness)
                                        xxxi = 1;
                                    else
                                        xxxi = 2;
                                } // hakan's edit
                            }
                            else
                            {
                                printf("[%d]flag is not continuing optimal is: %d\n", myrank, is_optimal_solution);
                            }

                            MPI_Allgather(&xxxi,
                                          1,
                                          MPI_INT,
                                          recieveExitStatus,
                                          1,
                                          MPI_INT,
                                          MPI_COMM_WORLD);

                            for (i = 0; i < size; i++)
                            {
                                if (recieveExitStatus[i] != 0)
                                    xxxi = recieveExitStatus[i];
                            }

                            if (xxxi)
                            {
                                xxxi = 0;
                                break;
                            }

                            Find_Best_Solution();
                        }

                        if (!is_optimal_solution)
                        { // is_optimal_solution is 1 if an optimal solution was printed before
                            clock_t end = clock();
                            TotalTime = (end - start) / (1000 * 1.0);
                            Find_Best_Solution();
                            printf("[%d]Optimal Found at iteration %d\n", myrank, generation);
                            WriteOutput();
                        }
                        else
                            printf("X[%d]Optimal Found at iteration %d\n", myrank, generation);

                    } // if instanceSignal
                }     // while instanceSignal
            }         // if configSignal
        }             // while configSignal
    }

#else // ifndef MPI_ON

    system("mkdir Solutions_GGA-CGT");
    system("mkdir Details_GGA-CGT");

    // READING EACH CONFIGURATION IN FILE "configurations.txt", CONTAINING THE PARAMETER VALUES FOR EACH EXPERIMENT
    if ((input_Configurations = fopen("Configurations.txt", "rt")) == NULL)
    {
        printf("\n INVALID FILE(configurations)\n");
        exit(1);
    }

    fgets(string, 150, input_Configurations);
    while (fgets(string, 150, input_Configurations))
    {
        line = string;
        token = strtok(line, "\t");
        conf = 1;
        token = strtok(line, "\t");
        P_size = 500;
        token = strtok(line, "\t");
        max_gen = 10000;
        token = strtok(line, "\t");
        p_m = 0.83;
        token = strtok(line, "\t");
        p_c = 0.2;
        token = strtok(line, "\t");
        k_ncs = 1.3;
        token = strtok(line, "\t");
        k_cs = 4;
        token = strtok(line, "\t");
        B_size = 0.1;
        token = strtok(line, "\t");
        life_span = 10;
        token = strtok(line, "\t");
        seed = 1;
        token = strtok(line, "\t");
        save_bestSolution = 0;

        printf("\nRunning With Configuration: %ld\n", conf);
        strcpy(nameC, "Solutions_GGA-CGT/GGA-CGT_(");
        snprintf(aux, sizeof(aux), "%ld", conf);
        strcat(nameC, aux);
        strcat(nameC, ").txt");
        output = fopen(nameC, "w+");
        fprintf(output, "CONF\t|P|\tmax_gen\tn_m\tn_c\tk1(non-cloned_solutions)\tk2(cloned_solutions)\t|B|\tlife_span\tseed");
        fprintf(output, "\n%ld\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\t%d", conf, P_size, max_gen, p_m, p_c, k_ncs, k_cs, B_size, life_span, seed);
        fprintf(output, "\nInstancias \t L2 \t Bins \t Gen \t Time");
        fclose(output);

        // READING FILE "instances.txt" CONTAINING THE NAME OF BPP INSTANCES TO PROCESS
        if ((input_Instances = fopen("instances.txt", "rt")) == NULL)
        {
            printf("\n INVALID FILE");
            exit(1);
        }

        while (fgets(file, 50, input_Instances))
        {
            line = file;
            token = strtok(line, "\n");
            *(line - 2) = '\0'; // to convert the newline character to null so that newline does not mess up the input

            LoadData();
            for (i = 0; i < number_items; i++)
                ordered_weight[i] = i;
            Sort_Descending_Weights(ordered_weight, number_items);
            LowerBound();
            seed_permutation = seed;
            seed_emptybin = seed;
            for (i = 0; i < P_size; i++)
            {
                ordered_population[i] = i;
                random_individuals[i] = i;
                best_individuals[i] = i;
            }
            Clean_population();
            is_optimal_solution = 0;
            generation = 0;
            for (i = 0, j = n_; j < number_items; i++)
                permutation[i] = ordered_weight[j++];
            repeated_fitness = 0;

            // procedure GGA-CGT
            start = clock();
            // long int count;
            if (!Generate_Initial_Population())
            { // Generate_Initial_Population() returns 1 if an optimal solution is found

                // printf("Number of items: %ld\n", number_items);
                // count = 0;
                // for (i = 0 ; i < number_items+4 ; i++ ) {
                //    count = count + population[ordered_population[1]][i].L.num;
                //}
                // printf("Linked list count: %ld\n", count);

                for (generation = 0; generation < max_gen; generation++)
                {
                    if (xxxi = Generation()) // Generation() returns 1 if an optimal solution was found
                        break;

                        //                   printf("[SEQ] gen: %dm xxxi %d\n", generation, xxxi);

#ifdef VERBOSE
                    cout << "Generation : " << generation << endl;
#else
                    if (generation % 100 == 0)
                        cout << "Generation : " << generation << " MAX_GEN: " << max_gen << endl;

#endif
                    Find_Best_Solution();
                }
            }

            printf("[SEQ] gen: %dm xxxi %d(L2: %ld - Best: %f)\n", generation, xxxi, L2, global_best_solution[number_items + 1].Bin_Fullness);
            if (!is_optimal_solution)
            { // is_optimal_solution is 1 if an optimal solution was printed before
                clock_t end = clock();
                TotalTime = (end - start) / (1000 * 1.0);
                Find_Best_Solution();
                printf("[SEQ]Optimal Found at iteration %d\n", generation);
                WriteOutput();
            }
            else
                printf("X[SEQ]Optimal Found at iteration %d\n", generation);
        }
        fclose(input_Instances);
    }
    fclose(input_Configurations);

#ifdef PRINT_POPULATION_CONTENTS
    FILE *popOutFile;

    popOutFile = fopen("population.txt", "w");
    if (!popOutFile)
    {
        cout << "Population file cannot be opened" << endl;
        exit(0);
    }

    printAllPopulation(*population, popOutFile, n_bins);
    fclose(popOutFile);
#endif

#endif // ifndef MPI_ON

#ifdef MPI_ON
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[%d] Finalized\n", myrank);
    MPI_Finalize();
#endif

    printf("\nEnd of process\n");
}