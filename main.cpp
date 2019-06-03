#define AMPI_RENAME_EXIT 0

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mpi.h>
#include <optional>
#include <sstream>

#include "CLI11.hpp"
#include "date.h"

using namespace std;

#define _unused(x) ((void)(x))

static double T_x_0_boundaryconditions(int xi, int nx)
{
    /*This is the boundary condition along the "bottom" of the grid, where y=0*/
    /*xi is the index of x*/
    auto xi_d = static_cast<double>(xi);
    auto nx_d = static_cast<double>(nx);

    auto cosine = cos((xi_d + 0.5) / nx_d * M_PI);
    return cosine * cosine;
}

static double T_x_pi_boundaryconditions(int xi, int nx)
{
    /*This is the boundary condition along the "top" of the grid, where y=pi*/
    /*xi is the index of x*/
    auto xi_d = static_cast<double>(xi);
    auto nx_d = static_cast<double>(nx);

    auto sine = sin((xi_d + 0.5) / nx_d * M_PI);
    return sine * sine;
}

static double** grid_creator(const size_t nx, const size_t n)
{
    /*Create the array to store the temperatures*/

    auto pointer = new double*[nx];
    for (size_t i = 0; i < nx; i++) {
        pointer[i] = new double[n];
    }

    return pointer;
}

static int stepper(double** T, double** T2, const int nx, const double dx, const double dt, const int ncols, const int rank)
{
    /*Step things forward one step in time.*/

    double adjacent[4];
    for (int i = 0; i < nx; i++) //which row, y
    {
        for (int j = 1; j < (ncols + 2 - 1); j++) //which column, x
        {
            if (i == 0) //corresponds to the top
            {
                adjacent[0] = T_x_pi_boundaryconditions((rank * ncols + j - 1), nx);
            } else {
                adjacent[0] = T[i - 1][j];
            }

            adjacent[1] = T[i][j + 1];

            if (i == nx - 1) //corresponds to the bottom
            {
                adjacent[2] = T_x_0_boundaryconditions((rank * ncols + j - 1), nx);
            } else {
                adjacent[2] = T[i + 1][j];
            }

            adjacent[3] = T[i][j - 1];

            T2[i][j] = T[i][j] + dt / (dx * dx) * (adjacent[0] + adjacent[1] + adjacent[2] + adjacent[3] - 4. * T[i][j]);
        }
    }

    return 0;
}

static string get_time(void)
{
    return date::format("[%T]", std::chrono::system_clock::now());
}

static void write_to_file(double** T_arr, int ncols, int nx, const string& outputfilename, bool append)
{
    auto mode = append ? ios_base::app : ios_base::trunc;
    ofstream ofile(outputfilename, mode);
    ofile.exceptions(fstream::failbit | fstream::badbit);
    if (!append) {
        // write the header
        ofile << "#Final temperature stored in grid format, MPI style\n"
              << "#Columns represent the rows in the actual grid and vice-versa.  This is so that each separate rank's file can easily be appended to the others to produce a full file.\n";
    }
    auto digits = numeric_limits<double>::digits10;

    //print the data to file
    for (int i = 1; i < (ncols + 2 - 1); i++) {
        for (int j = 0; j < nx; j++) {
            ofile << setprecision(digits) << fixed << T_arr[j][i] << "\t"; //For a fixed column all the elements get printed horizontally in the file.
        }
        ofile << "\n";
    }
}

static void read_from_file(double** T_arr, int ncols, int nx, const string& inputfile, int rank)
{
    ifstream ifile(inputfile);
    ifile.exceptions(fstream::failbit | fstream::badbit);
    // skip comments
    while (ifile.peek() == '#') {
        ifile.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    // ignore the lines for previous ranks
    for (int i = 0; i < ncols * rank; ++i) {
        ifile.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    // read the data
    for (int i = 1; i < (ncols + 2 - 1); i++) {
        for (int j = 0; j < nx; j++) {
            ifile >> T_arr[j][i]; //For a fixed column all the elements get printed horizontally in the file.
        }
    }
}

static void zeroize_matrix(double** T_arr, int ncols, int nx)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 1; j < (ncols + 2 - 1); j++) {
            T_arr[i][j] = 0.0;
        }
    }
}

int main(int argc, char* argv[])
{

    double start_time = MPI_Wtime(); //we're timing this run

    /*Initialize MPI*/
    int rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program.  Now quitting\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    int numtasks, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); //the number of tasks, number of processors to use
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //the rank of this process

    optional<int> req_steps;
    optional<string> initial;
    int nx;
    bool noresults = false;
    CLI::App app;
    app.add_option("--size", nx, "size of the grid")->required(true);
    app.add_option("--initial", initial, "inital conditions");
    app.add_option("--steps", req_steps, "steps to simulate");
    app.add_flag("--noresults", noresults, "don't write the results after the computation, do a purely profiling run");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    if (noresults && rank == 0) {
        cerr << "WARNING: results won't be written after the computation, as requested\n";
    }

    if (rank == 0 && nx % numtasks != 0) {
        /*Make sure the required grid size splits evenly between the number of processors*/
        printf("Your chosen nx does not divide nicely into your chosen number of threads.\n Change the code or your arguments.  Now quitting...\n");
        exit(1);
    }

    const int ncols = nx / numtasks; //The number of columns given to each processor

    MPI_Request reqs[4]; //checkers for use later during the actual MPI part
    MPI_Status stats[4];

    int check; /*used for checking outputs to make sure they are good*/
    double** T_arr; /*This will be a pointer to an array of pointers which will host the grid itself*/
    double** T_arr_2; /*This will be used to calculate the new T values at each time step while keeping the old values in T_arr for calculation purposes*/
    double** T_pointer_temp; /*To be used to switch the pointers T_arr and T_arr_2*/
    const double fraction_of_maximum_time_step = 0.8; /*This is the fraction of the largest numerically stable timestep, calculated below, that we want our dt to actually be.  Keeping it some sane fraction will allow us to get an exact fraction of the maximum time we want. In units of kappa*/
    const double dx = M_PI / static_cast<double>(nx); //physical size of grid cells
    const double dt = dx * dx / 4.0 * fraction_of_maximum_time_step; /*This is the time step size, in units of kappa, which later cancel*/
    const double tmax = (0.5 * M_PI * M_PI); //maximum time to run
    const int ntstep = req_steps.value_or(static_cast<int>(tmax / dt)); //number of time steps

    const int prev = (rank == 0) ? numtasks - 1 : rank - 1; //define the rank number of previous
    const int next = (rank == (numtasks - 1)) ? 0 : rank + 1; //and next processor

    const int tag1 = 1; //tags for the MPI
    const int tag2 = 2; //tags for the MPI

    const size_t nxsize = static_cast<size_t>(nx);

    T_arr = grid_creator(nxsize, static_cast<size_t>(ncols + 2)); //allocate memory for the temperature grids.
    T_arr_2 = grid_creator(nxsize, static_cast<size_t>(ncols + 2)); //We use ncols+2 to provide some ghost cells for holding the boundary temperature values

    //Now initialize the array to the initial conditions
    //Our initial conditions are to have T=0 everywhere

    if (initial.has_value()) {
        string& file = initial.value();
        try {
            read_from_file(T_arr, ncols, nx, file, rank);
        } catch (const exception& e) {
            cerr << "Error reading file " << file << ": " << e.what() << "\n";
            exit(1);
        }
    } else {
        zeroize_matrix(T_arr, ncols, nx);
    }

    double* right_pass = new double[nxsize];
    double* left_pass = new double[nxsize];
    double* right_accept = new double[nxsize];
    double* left_accept = new double[nxsize];

#ifdef AMPI
    AMPI_Register_just_migrated([] {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        cout << "Rank " << rank << "migrated\n";
    });
#endif

    for (int i = 0; i < ntstep; i++) {
        if (rank == 0 && ntstep >= 200 && i % (ntstep / 200) == 0) {
            double progress = 100.0 * i / static_cast<double>(ntstep);
            printf("%s Progress: %.1f%%"
                   ". Computing step %d/%d\n",
                get_time().c_str(), progress, i, ntstep);
        }

        //pass the boundary columns between the processes
        for (int l = 0; l < nx; l++) {
            left_pass[l] = T_arr[l][1]; //pass left side of array to previous processor
            right_pass[l] = T_arr[l][ncols + 2 - 1 - 1]; //pass the right side of array to next processor
                //the index is given by: ncols+2 total number of columns, then -1 because zero indexed, then -1 again because we want to pass the second to last column.
        }

        /*MPI code to send and receive the appropriate boundary cells*/
        MPI_Isend(left_pass, nx, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(right_pass, nx, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[1]);

        MPI_Irecv(left_accept, nx, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(right_accept, nx, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[3]);

        /*Wait until all necessary memory is sent and received*/
        MPI_Waitall(4, reqs, stats);

        for (int l = 0; l < nx; l++) {
            T_arr[l][0] = left_accept[l];
            T_arr[l][ncols + 2 - 1] = right_accept[l];

            //This tests to make sure that the initial conditions all got passed correctly, i.e., there are no non-zero cells being received.
            if (i == 0) {
                if (left_accept[l] > 1.e-12)
                    printf("Offending leftaccept: index %d,  rank %d\n value: %e", l, rank, left_accept[l]);
                if (right_accept[l] > 1.e-12)
                    printf("Offending rightaccept: index %d,  rank %d\n value: %e", l, rank, right_accept[l]);
            }
        }

        check = stepper(T_arr, T_arr_2, nx, dx, dt, ncols, rank); //step forward one time step
        assert(check == 0);
        _unused(check);

        /*The following switches the pointers T_arr and T_arr_2, making T_arr now equal to the newly updated array formerly pointed to by T_arr_2 and giving the T_arr_2 pointer the old array*/
        T_pointer_temp = T_arr_2;
        T_arr_2 = T_arr;
        T_arr = T_pointer_temp;
#ifdef AMPI
        if (i % 10 == 0) {
            cout << "Trying to migrate...\n";
            AMPI_Migrate(AMPI_INFO_LB_ASYNC);
        }
#endif
    }

    //Print how long this run took, for this rank
    printf("numtasks: %d, rank: %d, nx: %d, time: %lf\n", numtasks, rank, nx, MPI_Wtime() - start_time);

    // Write the outputs
    if (!noresults) {
        ostringstream fname_ss;
        fname_ss << "heat_mpi." << nx << "." << numtasks << ".output.dat";
        string filename = fname_ss.str();

        if (rank == 0) {
            try {
                write_to_file(T_arr, ncols, nx, filename, false);
                for (int rank2 = 1; rank2 < numtasks; ++rank2) {
                    for (int i = 0; i < nx; ++i) {
                        MPI_Recv(T_arr[i], ncols, MPI_DOUBLE, rank2, 0, MPI_COMM_WORLD, nullptr);
                    }
                    write_to_file(T_arr, ncols, nx, filename, true);
                }
            } catch (const exception& e) {
                cerr << "Error reading file " << filename << ": " << e.what() << "\n";
                exit(1);
            }

        } else {
            for (int i = 0; i < nx; ++i) {
                MPI_Send(T_arr[i], ncols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        if (rank == 0) {
            cout << "Skipped writing the results, as requested\n";
        }
    }

    MPI_Finalize();

    return 0;
}
