import pytest

from aethergraph.observability import (
    ObservabilityIndexRequiredError,
    ObservabilityUnavailableError,
    observability_storage_errors,
)
from aethergraph.storage.contracts import StorageError, StorageMigrationRequiredError


@pytest.mark.parametrize(
    "provider_error,public_error",
    [
        (StorageError, ObservabilityUnavailableError),
        (StorageMigrationRequiredError, ObservabilityIndexRequiredError),
    ],
)
def test_composed_inspection_translates_provider_errors(provider_error, public_error):
    error = provider_error("private provider path")
    with pytest.raises(public_error) as failure, observability_storage_errors():
        raise error
    assert failure.value.__cause__ is error
    assert "private provider path" not in str(failure.value)


def test_boundary_preserves_validation_and_success():
    with observability_storage_errors():
        result = 42
    assert result == 42
    error = ValueError("bad filter")
    with pytest.raises(ValueError) as failure, observability_storage_errors():
        raise error
    assert failure.value is error
