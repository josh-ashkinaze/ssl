import json
import random
from collections import deque
from pathlib import Path

from joblib import Parallel, delayed  # For parallel processing
from tqdm.auto import tqdm  # For progress bars


class JsonUtils():
    """
    A utility class for processing and filtering JSON Lines files.
    It supports sampling, filtering, and parallel processing of JSON entries,
    allowing users to provide their own custom validation logic.

    Example usage:

    utils = JsonUtils()

    # Now imagine this is a validator function
    def check_item_properties(entry, required_word=:
        try
            if entry['text'] == required_word:
                return True
        except:
            return false
        return False

    # And we init a specific validator to see if something contains the word "hello"
    contains_hello = partial(check_item_properties, required_word="hello")

    # Now we pass the validator to the read_and_filter_json_file method
    all_valid_items = utils.read_and_filter_json_file(
        "your_path_to_json.json",
        validator_function=contains_hello,
        n_jobs=-1,
        batch_size=3
    )

    > returns a list of all valid JSON objects (dictionaries) that contain the word "hello" in their 'text' field.

    """

    def __init__(self):
        """
        Initializes the JsonUtils class.
        """
        pass

    @staticmethod
    def _mp_process_line_for_filtering(line_validator_tuple):
        """
        Parses a JSON line and applies a user-provided validator.
        Designed for use with multiprocessing/joblib.

        This method is intended as a helper for parallel processing and typically
        should not be called directly by the user.

        Args:
            line_validator_tuple (tuple): A tuple containing:
                (line_content_string, validator_function).
                validator_function (callable): A function that takes a parsed JSON entry (dict)
                                               and returns True if valid, False otherwise. Can be None.

        Returns:
            object: The parsed JSON object (dict) if it's valid according to the
                    validator_function (or if validator_function is None and parsing succeeds),
                    otherwise None.
        """
        line_content, validator_func = line_validator_tuple
        if not line_content.strip():  # Skip empty or whitespace-only lines
            return None
        try:
            entry = json.loads(line_content)
            if validator_func is None or validator_func(entry):
                return entry  # Parsed and validated (or no validator)
            return None  # Failed validation
        except json.JSONDecodeError:
            tqdm.write(f"Warning: Could not parse line: {line_content[:100]}")
            return None  # Failed parsing

    def sample_jsons_from_file(self, filename, n, method='first', seed=42, validator_function=None):
        """
        Samples N entries from a JSON Lines file according to the specified method,
        using a user-provided validation function.

        Args:
            filename (str or Path): The path to the JSON Lines file.
            n (int): The number of entries to sample. If 0, an empty list is returned.
            method (str, optional): The sampling method. Must be one of:
                                    'first': Takes the first N valid entries.
                                    'last': Takes the last N valid entries.
                                    'random': Performs reservoir sampling for N random valid entries.
                                    Defaults to 'first'.
            seed (int, optional): The random seed for 'random' sampling. Defaults to 42.
            validator_function (callable, optional): A function that takes a JSON entry (dict)
                                                     and returns True if valid, False otherwise.
                                                     If None, all successfully parsed entries are considered valid.
                                                     Defaults to None.

        Returns:
            list: A list of sampled JSON objects (dictionaries).

        Raises:
            ValueError: If the method is not 'first', 'last', or 'random'.
            FileNotFoundError: If the filename does not exist.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Error: File '{filename}' not found.")
        if n == 0:
            return []

        samples = []
        if method == 'first':
            with open(filename, 'r', encoding='utf-8') as f:
                pbar = tqdm(desc=f"Finding first {n} valid entries from '{Path(filename).name}'", total=n, unit="entry")
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        if validator_function is None or validator_function(entry):
                            samples.append(entry)
                            pbar.update(1)
                            if len(samples) >= n:
                                break
                    except json.JSONDecodeError:
                        tqdm.write(f"Skipping unparseable line in 'first' method: {line[:100]}")
                pbar.close()
                if len(samples) < n:
                    tqdm.write(
                        f"Warning: Found only {len(samples)} valid entries of {n} requested using 'first' method.")
            return samples
        elif method == 'last':
            samples_deque = deque(maxlen=n)
            total_lines = None
            try:
                with open(filename, 'r', encoding='utf-8') as f_count:
                    total_lines = sum(1 for _ in f_count)
            except (IOError, OSError):
                pass

            with open(filename, 'r', encoding='utf-8') as f:
                with tqdm(desc=f"Scanning for last {n} valid entries from '{Path(filename).name}'", total=total_lines,
                          unit="line") as pbar:
                    for line in f:
                        pbar.update(1)
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            if validator_function is None or validator_function(entry):
                                samples_deque.append(entry)
                        except json.JSONDecodeError:
                            tqdm.write(f"Skipping unparseable line in 'last' method: {line[:100]}")
            if len(samples_deque) < n and n > 0:
                tqdm.write(
                    f"Warning: Found only {len(samples_deque)} valid entries of {n} requested using 'last' method.")
            return list(samples_deque)
        elif method == 'random':
            random.seed(seed)
            valid_items_seen = 0
            total_lines = None
            try:
                with open(filename, 'r', encoding='utf-8') as f_count:
                    total_lines = sum(1 for _ in f_count)
            except (IOError, OSError):
                pass

            with open(filename, 'r', encoding='utf-8') as f:
                with tqdm(desc=f"Random sampling for {n} valid entries from '{Path(filename).name}'", total=total_lines,
                          unit="line") as pbar:
                    for line in f:
                        pbar.update(1)
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            if validator_function is None or validator_function(entry):
                                valid_items_seen += 1
                                if len(samples) < n:
                                    samples.append(entry)
                                else:
                                    j = random.randint(0, valid_items_seen - 1)
                                    if j < n:
                                        samples[j] = entry
                        except json.JSONDecodeError:
                            tqdm.write(f"Skipping unparseable line in 'random' method: {line[:100]}")
                if len(samples) < n:
                    tqdm.write(f"Warning: Found only {len(samples)} valid random entries of {n} requested.")
            return samples
        else:
            raise ValueError("Method must be 'first', 'last', or 'random'")

    def filter_json_list_in_memory(self, json_list, validator_function, n_jobs=None):
        """
        Filters a list of JSON objects (already in memory) in parallel
        using a user-provided validator function with joblib.

        Args:
            json_list (list): A list of dictionaries, where each is a parsed JSON object.
            validator_function (callable): A function that takes a JSON entry (dict) and returns True
                                           if valid, False otherwise. If None, an error might occur
                                           or it might return the original list depending on joblib's behavior
                                           with None callables; it's best to provide a function or handle
                                           the "no validation" case before calling this method.
            n_jobs (int, optional): The number of CPU cores for parallel processing.
                                    -1 means using all available CPUs.
                                    1 means sequential processing.
                                    Defaults to -1.

        Returns:
            list: A new list containing only JSON objects that passed the validator_function.
        """
        if not json_list:
            return []
        if validator_function is None:
            tqdm.write(
                "Warning: No validator_function provided to filter_json_list_in_memory. Returning a copy of the list.")
            return list(json_list)

        if n_jobs is None:
            n_jobs = -1

        tasks = (delayed(validator_function)(item) for item in json_list)
        is_valid_results = Parallel(n_jobs=n_jobs, verbose=0)(tasks)

        validated_items = [item for item, is_valid_flag in zip(json_list, is_valid_results) if is_valid_flag]
        return validated_items

    def read_and_filter_json_file(self, filename, validator_function, n_jobs=None, batch_size=2048):
        """
        Reads a JSON Lines file, filters entries in parallel using a user-provided
        validator_function and joblib, returning a list of valid JSON objects.
        This method is memory-efficient for large files by processing in batches.

        Args:
            filename (str or Path): The path to the JSON Lines file.
            validator_function (callable): A function that takes a JSON entry (dict) and returns True
                                           if valid, False otherwise. If None, all parseable entries
                                           are returned.
            n_jobs (int, optional): Number of CPU cores for parallel processing.
                                    -1 for all CPUs, 1 for sequential. Defaults to -1.
            batch_size (int, optional): Number of lines per parallel batch. Defaults to 2048.

        Returns:
            list: A list of validated JSON objects (dictionaries).

        Raises:
            FileNotFoundError: If the filename does not exist.
        """
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Error: File '{filename}' not found.")

        if n_jobs is None:
            n_jobs = -1

        validated_jsons = []
        total_lines = None
        try:
            with open(filename, 'r', encoding='utf-8') as f_count:
                total_lines = sum(1 for _line in f_count)
        except (IOError, OSError) as e:
            tqdm.write(
                f"Info: Could not pre-count lines for '{Path(filename).name}' ({e}). Progress bar may be indeterminate.")

        pbar_desc = f"Reading & filtering '{Path(filename).name}' (joblib)"
        pbar = tqdm(total=total_lines, desc=pbar_desc, unit="line") if total_lines is not None else tqdm(desc=pbar_desc,
                                                                                                         unit="line")

        parallel_executor = Parallel(n_jobs=n_jobs, verbose=0)

        with open(filename, 'r', encoding='utf-8') as f:
            lines_batch_for_processing = []
            for line_content in f:
                lines_batch_for_processing.append((line_content, validator_function))
                if len(lines_batch_for_processing) >= batch_size:
                    try:
                        tasks = (delayed(JsonUtils._mp_process_line_for_filtering)(item_tuple)
                                 for item_tuple in lines_batch_for_processing)
                        batch_results = parallel_executor(tasks)
                        for entry in batch_results:
                            if entry:
                                validated_jsons.append(entry)
                    except Exception as e:
                        tqdm.write(f"Error processing a batch with joblib: {e}")
                    pbar.update(len(lines_batch_for_processing))
                    lines_batch_for_processing.clear()

            if lines_batch_for_processing:
                try:
                    tasks = (delayed(JsonUtils._mp_process_line_for_filtering)(item_tuple)
                             for item_tuple in lines_batch_for_processing)
                    batch_results = parallel_executor(tasks)
                    for entry in batch_results:
                        if entry:
                            validated_jsons.append(entry)
                except Exception as e:
                    tqdm.write(f"Error processing the final batch with joblib: {e}")
                pbar.update(len(lines_batch_for_processing))
        pbar.close()
        return validated_jsons
