import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from functools import partial
from collections import Counter, defaultdict
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEBUG_FILE_PATH = "/Users/wuhang/Desktop/cs/llm/stanford cs336/assignment1-basics/tests/debug_output.json"

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



# ## Usage
# with open(..., "rb") as f:
#     boundaries = find_chunk_boundaries(
#         f, num_processes, "<|endoftext|>".encode("utf-8"))
        
#     # The following is a serial implementation, but you can parallelize this 
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token

def pretokenization(special_tokens : list[str], chunk : str):
    # split the chunks by the special tokens
    pattern = "|".join(map(re.escape, special_tokens))
    splitted_chunks = re.split(pattern, chunk)
    # perform pre-tokenization in splitted chunks
    matches = []
    for splitted_chunk in splitted_chunks:
        matches += [match.group() for match in re.finditer(PAT, splitted_chunk)]
        
    word_counts = Counter(matches)
    return word_counts

def merge_pair(updator : tuple[bytes], max_freq_pair : tuple[bytes]) -> tuple[bytes]:
    new_updator = [bytes for bytes in updator]
    i = 0
    while i < len(new_updator)-1:
        if tuple((new_updator[i], new_updator[i+1])) == max_freq_pair:
            new_updator = new_updator[:i] + [max_freq_pair[0] + max_freq_pair[1]] + new_updator[i+2:]
            i += 1
        else:
            i += 1
    return tuple(new_updator)

        
def find_bytes_to_merge(successive_bytes : tuple[bytes], updator : tuple[bytes], pos : int):
    return tuple((updator[pos], updator[pos+1])) == successive_bytes

def bpe_tokenizer_training(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    num_processes = 8        
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<endoftext>".encode("utf-8")
        )
        # construct chunks
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end-start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        with Pool() as pool:
            word_counts_list = pool.map(partial(pretokenization, special_tokens), chunks)
        
        # merge word counts
        total_counts = Counter()
        for word_counts in word_counts_list:
            total_counts.update(word_counts)
            
        # convert str to tuple[bytes]
        # bytes_counts = Counter({word.encode("utf-8") : count for word, count in sorted_counts.items()})
        bytes_counter = Counter()
        for word, count in total_counts.items():
            bytes_tuple = tuple(char.encode("utf-8") for char in word)
            bytes_counter[bytes_tuple] = count
        
        # construct successive bytes pair and sum their frequency
        successive_bytes_counter = Counter()
        # construct a dict that records the updator of each pair of successive bytes
        update_dict = defaultdict(list) # dict[bytes, list[tuple[bytes]]]
        for bytes_tuple in bytes_counter.keys():
            for i in range(len(bytes_tuple)-1):
                # successive_bytes = b''.join(bytes_tuple[i:i+2])
                successive_bytes = tuple((bytes_tuple[i], bytes_tuple[i+1]))
                successive_bytes_counter[successive_bytes] += bytes_counter[bytes_tuple]
                if bytes_tuple not in update_dict[successive_bytes]: 
                    update_dict[successive_bytes] += [bytes_tuple]
                    
        # compute BPE merges
        merges = [] # list[tuple[bytes, bytes]]
        merge_times = vocab_size - len(special_tokens) - 256
        assert merge_times >= 0, "the vocab size should be greater than 256 + the number of special tokens"
        for _ in range(merge_times):
            # print(successive_bytes_counter)
            # exit(0)
            # with open(DEBUG_FILE_PATH, 'w') as f:
            #     debug_dict = {
            #         key.decode('utf-8')  : value 
            #         for key, value in successive_bytes_counter.items()
            #     }
            #     json.dump(debug_dict, f, indent=2)

            # find the bytes pair that has the greatest frequency lexicographically
            max_freq = max(successive_bytes_counter.values())
            max_keys = [k for k, v in successive_bytes_counter.items() if v == max_freq]
            max_freq_successive_bytes_sorted = max(max_keys)
            merges.append(max_freq_successive_bytes_sorted)
            max_freq_successive_bytes_sorted_merged = b''.join(max_freq_successive_bytes_sorted)
            # for successive_bytes, counts in successive_bytes_counter.items():
            # merge
            updators = update_dict[max_freq_successive_bytes_sorted] # list[tuple[bytes]]
            
            # construct key mapping
            key_mappings = {}
            for updator in updators:
                # new_key = []
                i = 0
                # merge the successive bytes first
                new_updator = merge_pair(updator, max_freq_successive_bytes_sorted)
                # compute successive byte to decrease
                successive_bytes_updated = defaultdict(bool) # {bytes : bool}
                while i < len(updator) - 1:
                    # a dict to record which bytes pair has been udpated
                    if find_bytes_to_merge(max_freq_successive_bytes_sorted, updator, i):
                        # notice this following new-merge-appending procedure will only happen once (even if there are different updators, after careful analysis of the merging, the merges for different updators are the same) 
                        # new_merge = tuple((updator[i], updator[i+1]))
                        # if new_merge not in merges:
                        #     merges.append(new_merge)
                        # new_key.append(max_freq_successive_bytes_sorted)
                        if i > 0 and not successive_bytes_updated[i-1]:
                            # new_successive_bytes_tuple = updator[i-1] + max_freq_successive_bytes_sorted
                            # successive_bytes_counter[new_successive_bytes_tuple] += bytes_counter[updator]
                            successive_bytes_updated[i-1] = True
                            successive_bytes_counter[tuple((updator[i-1], updator[i]))] -= bytes_counter[updator]
                            # if updator not in update_dict[new_successive_bytes_tuple]: update_dict[new_successive_bytes_tuple].append(updator)
                        if i < len(updator)-2 and not successive_bytes_updated[i+1]:
                            # new_successive_bytes_tuple = max_freq_successive_bytes_sorted + updator[i+2]
                            # successive_bytes_counter[new_successive_bytes_tuple] += bytes_counter[updator]
                            successive_bytes_updated[i+1] = True
                            successive_bytes_counter[tuple((updator[i+1], updator[i+2]))] -= bytes_counter[updator]
                            # if updator not in update_dict[new_successive_bytes_tuple]: update_dict[new_successive_bytes_tuple].append(updator)
                        i += 2
                    else:
                        # new_key.append(updator[i])
                        i += 1
                # add the final character which is not covered in the new_key
                # if (i == len(updator)-1):
                    # new_key.append(updator[-1])
                
                i = 0
                # compute successive bytes to increase
                while i < len(new_updator)-1:
                    if new_updator[i] == max_freq_successive_bytes_sorted_merged:
                        new_successive_bytes = tuple((max_freq_successive_bytes_sorted_merged, new_updator[i+1]))
                        successive_bytes_counter[new_successive_bytes] += bytes_counter[updator]
                        if updator not in update_dict[new_successive_bytes]:
                            update_dict[new_successive_bytes].append(updator)
                    elif new_updator[i+1] == max_freq_successive_bytes_sorted_merged:
                        new_successive_bytes = tuple((new_updator[i], max_freq_successive_bytes_sorted_merged))
                        successive_bytes_counter[new_successive_bytes] += bytes_counter[updator]
                        if updator not in update_dict[new_successive_bytes]:
                            update_dict[new_successive_bytes].append(updator)
                    i += 1
                key_mappings[updator] = tuple(new_updator)
            # after iterating through all the updators, pop the newly merged byte pairs in the counter and update_dict
            successive_bytes_counter.pop(max_freq_successive_bytes_sorted)    
            update_dict.pop(max_freq_successive_bytes_sorted)
            bytes_counter = Counter({key_mappings.get(k, k): v for k, v in bytes_counter.items()})
            update_dict = defaultdict(list, {key : [key_mappings.get(bytes_tuple, bytes_tuple) for bytes_tuple in tuple_list] for key, tuple_list in update_dict.items()})
            
        # construct the vocabulary
        basic_bytes = [bytes([i]) for i in range(256)]
        special_tokens = [special_token.encode("utf-8") for special_token in special_tokens]
        # merged_bytes = [key for key in successive_bytes_counter.keys() if key not in basic_bytes and key not in special_tokens]
        merged_bytes = [pair[0]+pair[1] for pair in merges]
        combined_tokens = basic_bytes + special_tokens + merged_bytes
        vocabulary = {idx : token for idx, token in enumerate(combined_tokens)} # dict[int, bytes]
        return vocabulary, merges

            
