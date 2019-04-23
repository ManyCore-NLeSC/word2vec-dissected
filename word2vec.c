//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// the maximum size of a string anywhere used in the program
#define MAX_STRING 100

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

// the size of the hashmap with indices to the words in the vocabulary
const int hashmap_indices_vocab_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

// struct for a word in the vocabulary
struct vocab_word {
  // the number a word occurs in the text
  long long count;
  // the word itself
  char *word;
};

// the input and output file names
char train_file[MAX_STRING], output_file[MAX_STRING];

// the file names where the vocabularies are saved and read from
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

// the vocabulary
struct vocab_word *vocab;

// parameters, see the main method for an explanation
int debug_mode = 2, window = 5, min_count = 5;

// parameter that controls removing infrequent words
int min_reduce = 1;

// hashmap that contains the indices of words into the vocabulary
int *hashmap_indices_vocab;

// The maximum size of the vocabulary.  This value will be 
// increased over time during a run.
long long vocab_max_size = 1000;
// the current size of vocabulary (the number of words in the vocabulary)
long long vocab_size = 0;

// the size of the dimension of a vector
long long dim_size = 100;

// The number of words that is used for training.  It will be
// incremented during a run.
long long nr_words_for_training = 0;


real starting_alpha = 0.025;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  double nr_words_for_training_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) nr_words_for_training_pow += pow(vocab[a].count, power);
  i = 0;
  d1 = pow(vocab[i].count, power) / nr_words_for_training_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].count, power) / nr_words_for_training_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = fgetc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % hashmap_indices_vocab_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (hashmap_indices_vocab[hash] == -1) return -1;
    if (!strcmp(word, vocab[hashmap_indices_vocab[hash]].word)) return hashmap_indices_vocab[hash];
    hash = (hash + 1) % hashmap_indices_vocab_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].count = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (hashmap_indices_vocab[hash] != -1) hash = (hash + 1) % hashmap_indices_vocab_size;
  hashmap_indices_vocab[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->count - ((struct vocab_word *)a)->count;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < hashmap_indices_vocab_size; a++) hashmap_indices_vocab[a] = -1;
  size = vocab_size;
  nr_words_for_training = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].count < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (hashmap_indices_vocab[hash] != -1) hash = (hash + 1) % hashmap_indices_vocab_size;
      hashmap_indices_vocab[hash] = a;
      nr_words_for_training += vocab[a].count;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].count > min_reduce) {
    vocab[b].count = vocab[a].count;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < hashmap_indices_vocab_size; a++) hashmap_indices_vocab[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (hashmap_indices_vocab[hash] != -1) hash = (hash + 1) % hashmap_indices_vocab_size;
    hashmap_indices_vocab[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING], eof = 0;
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < hashmap_indices_vocab_size; a++) hashmap_indices_vocab[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    nr_words_for_training++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", nr_words_for_training / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].count = 1;
    } else vocab[i].count++;
    if (vocab_size > hashmap_indices_vocab_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", nr_words_for_training);
  }
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].count);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c, eof = 0;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < hashmap_indices_vocab_size; a++) hashmap_indices_vocab[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin, &eof);
    if (eof) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].count, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", nr_words_for_training);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * dim_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * dim_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < dim_size; b++)
     syn1neg[a * dim_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < dim_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * dim_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dim_size;
  }
}

void TrainModelThread() {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;

  // The total number of words counted this far
  long long total_word_count = 0;

  unsigned long long next_random = 0;
  char eof = 0;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(dim_size, sizeof(real));
  real *neu1e = (real *)calloc(dim_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  real alpha = starting_alpha;

  fseek(fi, 0, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      total_word_count += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         total_word_count / (real)(nr_words_for_training + 1) * 100,
         total_word_count / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - total_word_count / (real)(nr_words_for_training + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, &eof);
        if (eof) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof || (word_count > nr_words_for_training)) {
      total_word_count += word_count - last_word_count;
      break;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < dim_size; c++) neu1[c] = 0;
    for (c = 0; c < dim_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1) continue;
      l1 = last_word * dim_size;
      for (c = 0; c < dim_size; c++) neu1e[c] = 0;
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * dim_size;
        f = 0;
        for (c = 0; c < dim_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < dim_size; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < dim_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
      }
      // Learn weights input -> hidden
      for (c = 0; c < dim_size; c++) syn0[c + l1] += neu1e[c];
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
}

void TrainModel() {
  long a, b;
  FILE *fo;
  printf("Starting training using file %s\n", train_file);
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  TrainModelThread();
  fo = fopen(output_file, "wb");
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, dim_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    for (b = 0; b < dim_size; b++) fprintf(fo, "%lf ", syn0[a * dim_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}


int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram\n");
    
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -negative 5\n\n");
    return 0;
  }

  // set the filenames to the null character, so to an empty string
  output_file[0] = '\0';
  save_vocab_file[0] = '\0';
  read_vocab_file[0] = '\0';
  
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) starting_alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

  // Allocate the vocabulary.  The size will grow during the run of the program
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));

  // alocate the hash table with indices
  hashmap_indices_vocab = (int *)calloc(hashmap_indices_vocab_size, sizeof(int));

  // exponent table for gradient descent
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
