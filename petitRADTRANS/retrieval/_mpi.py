from typing import Any

from petitRADTRANS.utils import mpi_launch_detected

MPI = None
comm = None
rank = 0
_mpi_checked = False


def get_mpi_context() -> tuple[Any, Any, int]:
    """Return ``(MPI, comm, rank)``, or ``(None, None, 0)`` when not running under MPI.

    mpi4py is only imported in processes an MPI launcher started (see
    :func:`petitRADTRANS.utils.mpi_launch_detected`). Any other process -- e.g. a
    jaxns_distributed worker, which rebuilds the retrieval and so reaches this
    function through RetrievalConfig -- takes the no-MPI path instead of running
    MPI_Init, which can abort it. ``PETITRADTRANS_DISABLE_MPI=1`` forces that path.
    """
    global MPI, comm, rank, _mpi_checked

    if not _mpi_checked:
        _mpi_checked = True

        imported_mpi = None
        if mpi_launch_detected():
            try:
                from mpi4py import MPI as imported_mpi
            except ImportError:
                imported_mpi = None

        if imported_mpi is None:
            MPI = None
            comm = None
            rank = 0
        else:
            MPI = imported_mpi
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

    return MPI, comm, rank
