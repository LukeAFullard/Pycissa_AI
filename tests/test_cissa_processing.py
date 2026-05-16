import numpy as np
import pytest
from pycissa.processing.cissa.cissa import Cissa, initial_data_checks

class TestInitialDataChecks:
    # Minimal valid x and use_32_bit for most tests
    default_x = np.array([0.0])
    default_use_32_bit = False

    def _get_x_for_t(self, t_array):
        if t_array.ndim == 0 or len(t_array) == 0: # scalar or empty
            return np.array([])
        return np.array([0.0] * len(t_array))

    # 1. Type Check Tests
    def test_type_check_mixed_non_numeric(self):
        t_test = np.array([1, 2, 'a', 3], dtype=object)
        x_test = self._get_x_for_t(t_test)
        with pytest.raises(TypeError) as excinfo:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        assert "Time vector t contains mixed data types" in str(excinfo.value)
        assert "<class 'str'>" in str(excinfo.value)
        assert "<class 'int'>" in str(excinfo.value)


    def test_type_check_mixed_numeric(self):
        t_test = np.array([1, 2.0, 3, 4.5]) # numpy will make this float64
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except TypeError:
            pytest.fail("TypeError raised unexpectedly for mixed numeric types.")

    def test_type_check_all_int(self):
        t_test = np.array([1, 2, 3])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except TypeError:
            pytest.fail("TypeError raised unexpectedly for all-int types.")

    def test_type_check_all_float(self):
        t_test = np.array([1.0, 2.0, 3.0])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except TypeError:
            pytest.fail("TypeError raised unexpectedly for all-float types.")

    def test_type_check_empty(self):
        t_test = np.array([])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except (TypeError, ValueError):
            pytest.fail("Error raised unexpectedly for empty t array.")

    def test_type_check_single_element_numeric(self):
        t_test = np.array([1])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except (TypeError, ValueError):
            pytest.fail("Error raised unexpectedly for single numeric element t array.")

    def test_type_check_single_element_non_numeric(self):
        t_test = np.array(['a'], dtype=object)
        x_test = self._get_x_for_t(t_test)
        try:
            # This should pass type check as there's only one type, even if non-numeric.
            # Order check won't run if not numeric.
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except (TypeError, ValueError):
            pytest.fail("Error raised unexpectedly for single non-numeric element t array.")

    # 2. Order Check Tests
    def test_order_check_correctly_sorted(self):
        t_test = np.array([1, 2, 3, 4])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except ValueError:
            pytest.fail("ValueError raised unexpectedly for correctly sorted array.")

    def test_order_check_unsorted_middle(self):
        t_test = np.array([1, 3, 2, 4])
        x_test = self._get_x_for_t(t_test)
        with pytest.raises(ValueError) as excinfo:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        assert "Time vector t is not sorted in ascending order. Element 2 at index 2 is less than preceding element 3 at index 1." in str(excinfo.value)

    def test_order_check_unsorted_start(self):
        t_test = np.array([3, 1, 2, 4])
        x_test = self._get_x_for_t(t_test)
        with pytest.raises(ValueError) as excinfo:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        assert "Time vector t is not sorted in ascending order. Element 1 at index 1 is less than preceding element 3 at index 0." in str(excinfo.value)

    def test_order_check_unsorted_end(self):
        t_test = np.array([1, 2, 4, 3])
        x_test = self._get_x_for_t(t_test)
        with pytest.raises(ValueError) as excinfo:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        assert "Time vector t is not sorted in ascending order. Element 3 at index 3 is less than preceding element 4 at index 2." in str(excinfo.value)

    def test_order_check_reverse_sorted(self):
        t_test = np.array([4, 3, 2, 1])
        x_test = self._get_x_for_t(t_test)
        with pytest.raises(ValueError) as excinfo:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        assert "Time vector t is not sorted in ascending order. Element 3 at index 1 is less than preceding element 4 at index 0." in str(excinfo.value)

    def test_order_check_empty_array(self):
        t_test = np.array([])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except ValueError:
            pytest.fail("ValueError raised unexpectedly for empty array (order check).")

    def test_order_check_single_element_array(self):
        t_test = np.array([10])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except ValueError:
            pytest.fail("ValueError raised unexpectedly for single element array (order check).")

    def test_order_check_all_equal_elements(self):
        t_test = np.array([2, 2, 2, 2])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except ValueError:
            pytest.fail("ValueError raised unexpectedly for array with all equal elements.")

    def test_order_check_mixed_numeric_sorted(self):
        t_test = np.array([1, 2.0, 3, 4.5])
        x_test = self._get_x_for_t(t_test)
        try:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        except ValueError:
            pytest.fail("ValueError raised unexpectedly for sorted mixed numeric array.")

    def test_order_check_mixed_numeric_unsorted(self):
        t_test = np.array([1, 3.0, 2, 4.5]) # Numpy makes this [1., 3., 2., 4.5]
        x_test = self._get_x_for_t(t_test)
        with pytest.raises(ValueError) as excinfo:
            initial_data_checks(t_test, x_test, self.default_use_32_bit)
        # Original test case was 3.0 and 2. Numpy array becomes [1., 3., 2., 4.5]
        # So element 2 at index 2 is less than preceding element 3.0 at index 1.
        assert "Time vector t is not sorted in ascending order. Element 2.0 at index 2 is less than preceding element 3.0 at index 1." in str(excinfo.value) or \
               "Time vector t is not sorted in ascending order. Element 2 at index 2 is less than preceding element 3.0 at index 1." in str(excinfo.value)

    def test_initial_data_checks_datetime(self):
        """
        Tests for datetime64 handling in initial_data_checks.
        """
        use_32_bit = False # Not relevant for these t-focused tests

        # Test case 1: Successful conversion of list of date strings
        t_str_list = ['2023-01-01', '2023-01-02', '2023-01-03']
        x_dummy1 = np.array([1, 2, 3])
        t_processed, _ = initial_data_checks(np.array(t_str_list), x_dummy1, use_32_bit)
        assert t_processed.dtype == np.dtype('datetime64[ns]')
        expected_dt1 = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[ns]')
        assert np.array_equal(t_processed, expected_dt1)

        # Test case 2: Successful handling of an already sorted datetime64 array
        t_dt64_sorted = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[ns]')
        x_dummy2 = np.array([1, 2, 3])
        t_processed, _ = initial_data_checks(t_dt64_sorted, x_dummy2, use_32_bit)
        assert t_processed.dtype == np.dtype('datetime64[ns]')
        assert np.array_equal(t_processed, t_dt64_sorted)

        # Test case 3: ValueError for an unsorted datetime64 array
        t_dt64_unsorted = np.array(['2023-01-01', '2023-01-03', '2023-01-02'], dtype='datetime64[ns]')
        x_dummy3 = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Time vector t is not sorted in ascending order. Element 2023-01-02T00:00:00.000000000 at index 2 is less than preceding element 2023-01-03T00:00:00.000000000 at index 1."):
            initial_data_checks(t_dt64_unsorted, x_dummy3, use_32_bit)

        # Test case 4: ValueError for a list of strings that cannot be converted
        t_str_list_invalid = ['2023-01-01', 'not-a-date', '2023-01-03']
        x_dummy4 = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Time vector t could not be converted to datetime. Please ensure it is in a format convertible to datetime64."):
            initial_data_checks(np.array(t_str_list_invalid), x_dummy4, use_32_bit)

        # Test case 5: Numeric arrays are still handled correctly (sorted)
        t_numeric_sorted = np.array([1, 2, 3])
        x_dummy5 = np.array([1, 2, 3])
        t_processed_num_sorted, _ = initial_data_checks(t_numeric_sorted, x_dummy5, use_32_bit)
        assert np.array_equal(t_processed_num_sorted, t_numeric_sorted)

        # Test case 6: Numeric arrays are still handled correctly (unsorted)
        t_numeric_unsorted = np.array([1, 3, 2])
        x_dummy6 = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Time vector t is not sorted in ascending order. Element 2 at index 2 is less than preceding element 3 at index 1."):
            initial_data_checks(t_numeric_unsorted, x_dummy6, use_32_bit)

        # Test case 7: Single element datetime string
        t_str_single = ['2023-01-01']
        x_dummy7 = np.array([1])
        t_processed_single, _ = initial_data_checks(np.array(t_str_single), x_dummy7, use_32_bit)
        assert t_processed_single.dtype == np.dtype('datetime64[ns]')
        expected_dt_single = np.array(['2023-01-01'], dtype='datetime64[ns]')
        assert np.array_equal(t_processed_single, expected_dt_single)

        # Test case 8: Empty list (should not fail, handled by earlier checks or convert to empty datetime64)
        t_empty_list = []
        x_dummy8 = np.array([])
        # initial_data_checks will convert t to np.array([]) first.
        # The new logic is after `if not all(isinstance(item, (int, float, np.integer, np.floating)) for item in t):`
        # `all` on an empty sequence is True, so it skips the datetime conversion block.
        # This means it relies on previous numeric checks or passes through if empty.
        t_processed_empty, _ = initial_data_checks(np.array(t_empty_list), x_dummy8, use_32_bit)
        assert len(t_processed_empty) == 0
        # Depending on how np.array([]) is handled before datetime checks, its dtype might vary.
        # If it's float64 by default for empty numeric, it passes. If object, it might try conversion.
        # Let's assume it remains a simple empty array that passes existing numeric checks.
        # The key is it doesn't raise an error related to datetime processing.

# Existing tests below, ensure they are not overwritten if this is a partial update
# For this task, we are overwriting the whole file to include the new class at the top.

def test_cissa_init_use_32_bit_true_conversion():
    """
    Test 1: use_32_bit=True with convertible numerics.
    - Input x array with integers and float64.
    - Verify that all elements in x (as processed by initial_data_checks via Cissa instantiation) become np.float32.
    """
    t_test = np.array([1, 2, 3, 4])
    x_test_initial = [1, 2.0, np.int64(3), np.float64(4.0)]

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_test_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float32, "Array dtype should be float32"
    for val in x_processed:
        assert isinstance(val, np.float32), f"Value {val} is not float32"

def test_cissa_init_use_32_bit_false_no_conversion():
    """
    Test 2: use_32_bit=False (default).
    - Input x array with integers (e.g., np.int32) and np.float64.
    - Verify that the data types of elements in x remain unchanged (e.g. np.int32, np.float64).
    """
    t_test = np.array([1, 2, 3])

    # Test with integers
    x_int_initial = np.array([1, 2, 3], dtype=np.int32)
    cissa_int_instance = Cissa(t=np.copy(t_test), x=np.copy(x_int_initial), use_32_bit=False)
    x_int_processed = cissa_int_instance.x
    # Note: np.array([1,2,3], dtype=np.int32) when passed to np.array(x) in initial_data_checks
    # might become np.int64 by default if not handled carefully.
    # However, the key is it should NOT become float32.
    # The initial_data_checks function itself will try to make it a default numpy array (int64 or float64).
    # Let's check the type it becomes after initial_data_checks's np.array(x) call.
    # If initial_data_checks receives np.array([1,2,3], dtype=np.int32), it should keep it as is or convert to a default int type.
    # For this test, we expect the type to be preserved if it's already a NumPy array.
    assert x_int_processed.dtype == np.int32 or x_int_processed.dtype == np.int64, f"Integer array dtype {x_int_processed.dtype} was not expected int32 or int64"


    # Test with float64
    x_float64_initial = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cissa_float64_instance = Cissa(t=np.copy(t_test), x=np.copy(x_float64_initial), use_32_bit=False)
    x_float64_processed = cissa_float64_instance.x
    assert x_float64_processed.dtype == np.float64, "Float64 array dtype should remain float64"

    # Test with mixed Python types (which numpy will homogenize)
    x_mixed_initial = [1, 2.0, 3] # Will become float64 array
    cissa_mixed_instance = Cissa(t=np.copy(t_test), x=np.copy(x_mixed_initial), use_32_bit=False)
    x_mixed_processed = cissa_mixed_instance.x
    assert x_mixed_processed.dtype == np.float64, "Mixed Python types should lead to float64 numpy array"


def test_cissa_init_use_32_bit_true_mixed_handling():
    """
    Test 3: use_32_bit=True with mixed types including non-numeric and overflow.
    - Input x array with float64, convertible string, non-convertible string,
      a value that overflows float32, and a standard integer.
    - Verify correct conversions and retentions.
    """
    t_test = np.array([1, 2, 3, 4, 5])
    # Value that results in np.inf when converted to float32
    overflow_val = np.finfo(np.float64).max

    x_test_initial = np.array([1.0, "2.0", "text", 300, overflow_val], dtype=object)

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_test_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    # Assertions
    assert x_processed.dtype == object, "Array dtype should be object due to mixed types"

    assert isinstance(x_processed[0], np.float32), "Value 1.0 should be float32"
    assert x_processed[0] == np.float32(1.0), "Value 1.0 incorrect"

    assert isinstance(x_processed[1], np.float32), "Value '2.0' should be float32"
    assert x_processed[1] == np.float32(2.0), "Value '2.0' incorrect"

    assert isinstance(x_processed[2], str), "Value 'text' should remain a string"
    assert x_processed[2] == "text", "Value 'text' incorrect"

    assert isinstance(x_processed[3], np.float32), "Value 300 should be float32"
    assert x_processed[3] == np.float32(300.0), "Value 300 incorrect"

    # np.float32(np.finfo(np.float64).max) results in np.inf (float32 type)
    assert isinstance(x_processed[4], np.float32), "Overflow value should be float32 (inf)"
    assert x_processed[4] == np.float32(np.inf), "Overflow value should be inf"

def test_cissa_init_use_32_bit_true_overflow_within_float32_max_edge_case():
    """
    Test with a large number that is just at the edge of float32's max representable value.
    And one that is np.finfo(np.float32).max.
    """
    t_test = np.array([1, 2])
    # Max finite float32 value
    float32_max = np.finfo(np.float32).max

    x_test_initial = np.array([float32_max, float32_max * 0.999], dtype=np.float64) # Start as float64

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_test_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float32, "Array dtype should be float32"

    assert isinstance(x_processed[0], np.float32), "float32_max should be float32"
    assert x_processed[0] == np.float32(float32_max), "float32_max value incorrect"

    assert isinstance(x_processed[1], np.float32), "Slightly less than float32_max should be float32"
    # Comparison for floating point numbers requires tolerance or careful handling
    # For this specific case, direct conversion should be exact if it's representable
    assert x_processed[1] == np.float32(float32_max * 0.999)

def test_cissa_init_use_32_bit_true_with_standard_list_input():
    """
    Test with a standard Python list as input for x.
    """
    t_test = np.array([1, 2, 3])
    x_list_initial = [10, 20.5, 30] # Python ints and floats

    cissa_instance = Cissa(t=np.copy(t_test), x=x_list_initial, use_32_bit=True)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float32
    assert all(isinstance(val, np.float32) for val in x_processed)
    assert np.array_equal(x_processed, np.array([10.0, 20.5, 30.0], dtype=np.float32))

def test_cissa_init_use_32_bit_false_with_standard_list_input():
    """
    Test with a standard Python list as input for x and use_32_bit=False.
    """
    t_test = np.array([1, 2, 3])
    x_list_initial = [10, 20.5, 30] # Python ints and floats

    # Numpy will convert [10, 20.5, 30] to a float64 array by default.
    cissa_instance = Cissa(t=np.copy(t_test), x=x_list_initial, use_32_bit=False)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.float64
    assert np.array_equal(x_processed, np.array([10.0, 20.5, 30.0], dtype=np.float64))

def test_cissa_init_empty_input_array_with_use_32_bit_true():
    """
    Test with an empty input array for x and use_32_bit=True.
    """
    t_test = np.array([])
    x_empty_initial = np.array([])

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_empty_initial), use_32_bit=True)
    x_processed = cissa_instance.x

    # np.array([]).astype(np.float32) results in dtype float32.
    # An empty array when use_32_bit is true, after passing through np.empty_like(x, dtype=object)
    # and then x.astype(np.float32) might end up float64 if x was initially float64 (empty_like preserves type for data region)
    # or float32. Let's be more specific: initial_data_checks gets x = np.array([]).
    # new_x = np.empty_like(x, dtype=object) -> new_x is np.array([], dtype=object)
    # loop is skipped. x = new_x.
    # x.astype(np.float32) -> np.array([], dtype=np.float32).
    assert x_processed.dtype == np.float32
    assert len(x_processed) == 0

def test_cissa_init_empty_input_array_with_use_32_bit_false():
    """
    Test with an empty input array for x and use_32_bit=False.
    """
    t_test = np.array([])
    x_empty_initial = np.array([], dtype=np.int32) # Give a specific type

    cissa_instance = Cissa(t=np.copy(t_test), x=np.copy(x_empty_initial), use_32_bit=False)
    x_processed = cissa_instance.x

    assert x_processed.dtype == np.int32
    assert len(x_processed) == 0

def test_fit_raises_error_on_non_numeric_when_use_32_bit_true():
    """
    Test that fit() raises ValueError if x contains non-numeric data when use_32_bit=True.
    """
    t_test = np.array([1, 2, 3]) # Adjusted length to match x_test_initial for initial_data_checks
    x_test_initial = np.array([1.0, "cannot_convert", 3.0], dtype=object)

    # initial_data_checks is called during Cissa initialization.
    # The error related to x being non-numeric for fit is actually checked in initial_data_checks if use_32_bit=True
    # tries to convert to float32. If it remains object, fit() itself will try conversion.

    cissa_instance = Cissa(t=t_test, x=x_test_initial, use_32_bit=True)
    # At this point, cissa_instance.x is [np.float32(1.0), "cannot_convert", np.float32(3.0)] with dtype=object

    expected_error_msg = "All elements in the input array 'x' must be numeric or convertible to numeric type before fitting. Please check for non-numeric values."
    with pytest.raises(ValueError, match=expected_error_msg):
        cissa_instance.fit(L=2)


def test_pre_fill_gaps_use_32_bit_true():
    """
    Test pre_fill_gaps with use_32_bit=True ensures self.x becomes float32.
    """
    t_val = np.arange(5, dtype=np.float64)
    x_val = np.array([1.0, 2.0, np.nan, 4.0, np.nan], dtype=np.float64)

    cissa_instance = Cissa(t=t_val, x=x_val, use_32_bit=True)
    # After Cissa init, x_val (which contains NaNs) should be float32 if possible, or object.
    # np.array([1.0, np.nan], dtype=np.float64).astype(np.float32) is valid and results in a float32 array with NaNs.
    # So, initial_data_checks (when use_32_bit=True) should convert a float64 array with NaNs to float32 array with NaNs.
    assert cissa_instance.x.dtype == np.float32

    cissa_instance.pre_fill_gaps(L=2, estimate_error=False, component_selection_method='drop_smallest_n', test_repeats=0, verbose=False)

    assert cissa_instance.x.dtype == np.float32, "cissa_instance.x should be float32 after pre_fill_gaps with use_32_bit=True"
    assert not np.isnan(cissa_instance.x).any(), "NaNs should be filled in cissa_instance.x"


def test_pre_fill_gaps_use_32_bit_false():
    """
    Test pre_fill_gaps with use_32_bit=False ensures self.x remains float64.
    """
    t_val = np.arange(5, dtype=np.float64)
    x_val = np.array([1.0, 2.0, np.nan, 4.0, np.nan], dtype=np.float64)

    cissa_instance = Cissa(t=t_val, x=x_val, use_32_bit=False)
    assert cissa_instance.x.dtype == np.float64

    cissa_instance.pre_fill_gaps(L=2, estimate_error=False, component_selection_method='drop_smallest_n', test_repeats=0, verbose=False)

    assert cissa_instance.x.dtype == np.float64, "cissa_instance.x should be float64 after pre_fill_gaps with use_32_bit=False"
    assert not np.isnan(cissa_instance.x).any(), "NaNs should be filled in cissa_instance.x"


def test_fit_raises_error_on_non_numeric_when_use_32_bit_false():
    """
    Test that fit() raises ValueError if x contains non-numeric data when use_32_bit=False.
    """
    t_test = np.array([1, 2, 3]) # Adjusted length
    x_test_initial = np.array([1.0, "still_cannot_convert", 3.0], dtype=object)

    cissa_instance = Cissa(t=t_test, x=x_test_initial, use_32_bit=False)
    # With use_32_bit=False, cissa_instance.x will be the object array.

    expected_error_msg = "All elements in the input array 'x' must be numeric or convertible to numeric type before fitting. Please check for non-numeric values."
    with pytest.raises(ValueError, match=expected_error_msg):
        cissa_instance.fit(L=2)
