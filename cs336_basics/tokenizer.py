import magic
import json
from typing import Iterable, Iterator
from cs336_basics.bpe_tokenizer_train import PAT
import regex as re
class Tokenizer():
    def __init__(self, vocab : dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens : list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab = {v : k for k, v in self.vocab.items()} # a reverse vocabulary for encoding text
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token
    
    @classmethod
    def from_files(cls, vocab_filepath : str, merges_filepath : str, special_tokens : list[str] | None = None):
        mime = magic.Magic(mime=True)
        vocab_file_type = mime.from_file(vocab_filepath)
        merges_file_type = mime.from_file(merges_filepath)
        vocab = {}
        if vocab_file_type == "application/json":
            with open(vocab_filepath, "rb") as f:
                vocab = json.load(f)
        elif vocab_filepath == "text/plain":
            pass
        
        if merges_file_type == "application/json":
            with open(merges_filepath, "rb") as f:
                merges = json.load(f)
        elif merges_file_type == "text/plain":
            with open(merges_filepath, "r") as f:
                for line in f:
                    merged_pair = line.strip().split(' ')
                    merges.append(merged_pair)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text : str) -> list[int]:
        # TODO: deal with long input text
        splitted_words = self._pretokenize(text)
        token_ids = []
        for word in splitted_words:
            token_ids.extend(self._tokenize(word))
        return token_ids
    
    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
        
    def decode(self, ids : list[str]) -> str:
        output = b''
        for id in ids:
            output += self.vocab[id]
        return output.decode("utf-8", errors="replace")
    
    def _pretokenize(self, text : str) -> list[str]:
        splitted_text = [text] # list[str]
        # split the text by special tokens first
        if self.special_tokens is not None:
            for special_token in sorted(self.special_tokens, key=lambda x : -len(x)):
                new_splitted_text = []
                for chunk in splitted_text: # chunk : str
                    if chunk in self.special_tokens:
                        new_splitted_text += [chunk]
                        continue
                    new_chunk_list = re.split(re.escape(special_token), chunk) # list[str]
                    # append the special token after each splitted chunk
                    for i in range(len(new_chunk_list)):
                        new_splitted_text.append(new_chunk_list[i])
                        if i != len(new_chunk_list) - 1:
                            new_splitted_text += [special_token]
                splitted_text = new_splitted_text
            
        # split the text by PAT
        matches = []
        for splitted_chunk in splitted_text:
            if self.special_tokens is not None:
                if splitted_chunk not in self.special_tokens:
                    matches += [match.group() for match in re.finditer(PAT, splitted_chunk)]
                else:
                    matches += [splitted_chunk] # special token
            else:
                matches += [match.group() for match in re.finditer(PAT, splitted_chunk)]
            # matches += [match.group() for match in re.finditer(PAT, splitted_chunk)] if (self.special_tokens and splitted_chunk not in self.special_tokens) else [splitted_chunk]
        return matches

    def _tokenize(self, word : str) -> list[int]:
        if self.special_tokens is not None:
            if word in self.special_tokens:
                return [self.reverse_vocab[word.encode("utf-8")]]
        # convert word from str to bytes
        bytes_word = list([bytes([byte]) for char in word for byte in char.encode("utf-8")])
        if bytes_word == list():
            return []
        # try to merge successive bytes pair
        token_ids = []
        for merge_pair in self.merges:
            i = 0
            while i < len(bytes_word)-1:
                if merge_pair[0] == bytes_word[i] and merge_pair[1] == bytes_word[i+1]:
                    bytes_word = bytes_word[:i] + [bytes_word[i]+bytes_word[i+1]] + bytes_word[i+2:]
                i += 1
        for token in bytes_word:
            token_ids.append(self.reverse_vocab[token])
        return token_ids
        