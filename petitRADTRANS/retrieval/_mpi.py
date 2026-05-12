from typing import Any

MPI = None
comm = None
rank = 0
_mpi_checked = False


def get_mpi_context() -> tuple[Any, Any, int]:
    global MPI, comm, rank, _mpi_checked

    if not _mpi_checked:
        _mpi_checked = True

        try:
            from mpi4py import MPI as imported_mpi
        except ImportError:
            MPI = None
            comm = None
            rank = 0
        else:
            MPI = imported_mpi
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

    return MPI, comm, rank