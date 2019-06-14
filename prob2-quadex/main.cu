#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <utility>
#include <algorithm>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

const unsigned int alpha = 13;

using namespace std;
using namespace cuFIXNUM;

template< typename fixnum >
struct mul_and_convert {
  // redc may be worth trying over cios
  typedef modnum_monty_cios<fixnum> modnum;
  __device__ void operator()(fixnum &r, fixnum a, fixnum b, fixnum my_mod) {
      modnum mod = modnum(my_mod);

      fixnum sm;
      mod.mul(sm, a, b);

      fixnum s;
      mod.from_modnum(s, sm);

      r = s;
  }
};

template< typename fixnum >
struct add_num_convert {
	// redc may be worth trying over cios
   typedef modnum_monty_cios<fixnum> modnum;
    __device__ void operator()(fixnum &r, fixnum a, fixnum b, fixnum my_mod) {
      modnum mod = modnum(my_mod);

      fixnum sm;
      mod.add(sm, a, b);

      fixnum s;
      mod.from_modnum(s, sm);

      r = s;

  }
};

template< typename fixnum >
struct mul_scalar_convert {

  // redc may be worth trying over cios
  typedef modnum_monty_cios<fixnum> modnum;
 __device__ void operator()(fixnum &r, fixnum a, fixnum my_mod) {
      cuFIXNUM::warp_fixnum<128, cuFIXNUM::u64_fixnum> alpha_fixnum(alpha);
      modnum mod = modnum(my_mod);

      fixnum sm;
      mod.mul(sm, a, alpha_fixnum);

      fixnum s;
      mod.from_modnum(s, sm);

      r = s;

  }

// __device__ void operator()(fixnum &r0, fixnum &r1, fixnum a0,
//            fixnum a1, fixnum b0, fixnum b1, fixnum my_mod) {
// Logic: 
//  var a0_b0 = fq_mul(a.a0, b.a0);
//  var a1_b1 = fq_mul(a.a1, b.a1);
//  var a1_b0 = fq_mul(a.a1, b.a0);
//  var a0_b1 = fq_mul(a.a0, b.a1);
//  return {
//    a0: fq_add(a0_b0, fq_mul(a1_b1, alpha)),
//    a1: fq_add(a1_b0, a0_b1)
//  };
//     fixnum a0_b0;
//      mulFunc(a0_b0, a, b, my_mod);
//  }


};

template< int fn_bytes, typename fixnum_array >
void print_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);

    for (int i = 0; i < lrl; i++) {
      printf("%i ", local_results[i]);
    }
    printf("\n");
}

void print_uint8_array(uint8_t* array, int size) {
    for (int i = 0; i < size; i ++) {
        printf("%02x", array[i]);
    }
    printf("\n");
}

template< int fn_bytes, typename fixnum_array >
vector<uint8_t*> get_fixnum_array_dyn(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t *local_results = (uint8_t*) malloc(lrl*sizeof(uint8_t));
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);
    vector<uint8_t*> res_v;
    for (int n = 0; n < nelts; n++) {
      uint8_t* a = (uint8_t*)malloc(fn_bytes*sizeof(uint8_t));
      for (int i = 0; i < fn_bytes; i++) {
        a[i] = local_results[n*fn_bytes + i];
      }
      res_v.emplace_back(a);
    }
    free(local_results);
    return res_v;
}


template< int fn_bytes, typename fixnum_array >
vector<uint8_t*> get_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);
    vector<uint8_t*> res_v;
    for (int n = 0; n < nelts; n++) {
      uint8_t* a = (uint8_t*)malloc(fn_bytes*sizeof(uint8_t));
      for (int i = 0; i < fn_bytes; i++) {
        a[i] = local_results[n*fn_bytes + i];
      }
      res_v.emplace_back(a);
    }
    return res_v;
}

template< int fn_bytes, typename word_fixnum, template <typename> class Func >
pair<vector<uint8_t*>, vector<uint8_t*> >
 compute_quad_product(std::vector<uint8_t*> a_a0,
	       	std::vector<uint8_t*> a_a1,
	       	std::vector<uint8_t*> b_a0,
	       	std::vector<uint8_t*> b_a1, uint8_t* input_m_base) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    int nelts = a_a0.size();

    uint8_t *input_a_a0 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_a_a0[i] = a_a0[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_a_a1 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_a_a1[i] = a_a1[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_b_a0 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_b_a0[i] = b_a0[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_b_a1 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_b_a1[i] = b_a1[i/fn_bytes][i%fn_bytes];
    }


    uint8_t *input_m = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_m[i] = input_m_base[i%fn_bytes];
    }


    // TODO reuse modulus as a constant instead of passing in nelts times
    fixnum_array *res_a0_b0, *res_a1_b1, *res_a1_b0, *res_a0_b1, *res_a1_b1_alpha, *out_a0, *out_a1;
    fixnum_array *in_a_a0, *in_a_a1, *in_b_a0, *in_b_a1, *inM;
    in_a_a0 = fixnum_array::create(input_a_a0, fn_bytes * nelts, fn_bytes);
    in_a_a1 = fixnum_array::create(input_a_a1, fn_bytes * nelts, fn_bytes);
    in_b_a0 = fixnum_array::create(input_b_a0, fn_bytes * nelts, fn_bytes);
    in_b_a1 = fixnum_array::create(input_b_a1, fn_bytes * nelts, fn_bytes);
    inM = fixnum_array::create(input_m, fn_bytes * nelts, fn_bytes);
    res_a0_b0 = fixnum_array::create(nelts);
    res_a1_b1 = fixnum_array::create(nelts);
    res_a1_b0 = fixnum_array::create(nelts);
    res_a0_b1 = fixnum_array::create(nelts);
    res_a1_b1_alpha = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res_a0_b0, in_a_a0, in_b_a0, inM);
    fixnum_array::template map<Func>(res_a0_b1, in_a_a0, in_b_a1, inM);
    delete in_a_a0;

    fixnum_array::template map<Func>(res_a1_b1, in_a_a1, in_b_a1, inM);
    fixnum_array::template map<Func>(res_a1_b0, in_a_a1, in_b_a0, inM);


    fixnum_array::template map<mul_scalar_convert>(res_a1_b1_alpha, res_a1_b1, inM); 
    delete in_a_a1;
    delete in_b_a0;
    delete in_b_a1;
 
    out_a0 = fixnum_array::create(nelts);
    out_a1 = fixnum_array::create(nelts);

    fixnum_array::template map<add_num_convert>(out_a0, res_a1_b1_alpha, res_a0_b0, inM); 
    fixnum_array::template map<add_num_convert>(out_a1, res_a1_b0, res_a0_b0, inM); 



    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete inM;
    delete res_a0_b0;
    delete res_a1_b1;
    delete res_a0_b1;
    delete res_a1_b0;
    delete res_a1_b1_alpha;
    delete[] input_a_a0;
    delete[] input_a_a1;
    delete[] input_b_a0;
    delete[] input_b_a1;
    delete[] input_m;

    vector<uint8_t*> v_res_a0 = get_fixnum_array_dyn<fn_bytes, fixnum_array>(out_a0, nelts);
    vector<uint8_t*> v_res_a1 = get_fixnum_array_dyn<fn_bytes, fixnum_array>(out_a1, nelts);
    delete out_a0;
    delete out_a1;
    return make_pair(v_res_a0, v_res_a1);
}


template< int fn_bytes, typename word_fixnum, template <typename> class Func >
std::vector<uint8_t*> compute_product(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t* input_m_base) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    int nelts = a.size();

    uint8_t *input_a = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_a[i] = a[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_b = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_b[i] = b[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_m = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_m[i] = input_m_base[i%fn_bytes];
    }

    // TODO reuse modulus as a constant instead of passing in nelts times
    fixnum_array *res, *in_a, *in_b, *inM;
    in_a = fixnum_array::create(input_a, fn_bytes * nelts, fn_bytes);
    in_b = fixnum_array::create(input_b, fn_bytes * nelts, fn_bytes);
    inM = fixnum_array::create(input_m, fn_bytes * nelts, fn_bytes);
    res = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res, in_a, in_b, inM);

    vector<uint8_t*> v_res = get_fixnum_array<fn_bytes, fixnum_array>(res, nelts);

    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in_a;
    delete in_b;
    delete inM;
    delete res;
    delete[] input_a;
    delete[] input_b;
    delete[] input_m;
    return v_res;
}

uint8_t* read_mnt_fq(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)( buf + (bytes_per_elem - io_bytes_per_elem)), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

void write_mnt_fq(uint8_t* fq, FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  // mnt4_q
  uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // mnt6_q
  uint8_t mnt6_modulus[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");
  auto canon_results = fopen(argv[4], "r");

  size_t n;

   while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    printf("\n N = %d\n", n);
    if (elts_read == 0) { break; }

    std::vector<uint8_t*> x0_a0;
    for (size_t i = 0; i < n; ++i) {
      x0_a0.emplace_back(read_mnt_fq(inputs));
    }
    printf("\n read x0_a0\n");
    std::vector<uint8_t*> x0_a1;
    for (size_t i = 0; i < n; ++i) {
      x0_a1.emplace_back(read_mnt_fq(inputs));
    }
    printf("\n read x0_a1\n");

    std::vector<uint8_t*> y0_a0;
    for (size_t i = 0; i < n; ++i) {
      y0_a0.emplace_back(read_mnt_fq(inputs));
    }
    printf("\n read y0_a0\n");
    std::vector<uint8_t*> y0_a1;
    for (size_t i = 0; i < n; ++i) {
      y0_a1.emplace_back(read_mnt_fq(inputs));
    }
    printf("\n read y0_a1\n");


   printf("MNT4: \n");
   for (int i = 0; i < bytes_per_elem; i ++) {
        printf("%02x", mnt4_modulus[i]);
   }
   printf("\n");
  for (int i = 0; i < bytes_per_elem/2; i ++) {
        //std::swap(mnt4_modulus[i], mnt4_modulus[bytes_per_elem - i - 1]);
   }
   printf("After swapping MNT4: \n");
   for (int i = 0; i < bytes_per_elem; i ++) {
        printf("%02x", mnt4_modulus[i]);
   }
   printf("\n");

    std::pair<std::vector<uint8_t*>, std::vector<uint8_t*> > res_x
                    = compute_quad_product<bytes_per_elem, u64_fixnum, mul_and_convert>(x0_a0, x0_a1, y0_a0, y0_a1, mnt4_modulus);

    printf("\n compute_quad_product done.\n");


    for (size_t i = 0; i < n; ++i) {
      printf("cn: first ");
      print_uint8_array(read_mnt_fq(canon_results), io_bytes_per_elem);
      write_mnt_fq(res_x.first[i], outputs);
      printf("us: first ");
      print_uint8_array(res_x.first[i], bytes_per_elem);
      printf("cn: secon ");
      print_uint8_array(read_mnt_fq(canon_results), io_bytes_per_elem);
      write_mnt_fq(res_x.second[i], outputs);
      printf("us: secon ");
      print_uint8_array(res_x.second[i], bytes_per_elem);
    }
    printf("\n write finished.\n");

    for (size_t i = 0; i < n; ++i) {
      free(x0_a0[i]);
      free(x0_a1[i]);
      free(y0_a0[i]);
      free(y0_a1[i]);
      free(res_x.first[i]);
      free(res_x.second[i]);
      // free(res_x.first[i]);
      // free(res_x.second[i]);
    }

  }

  // diff 

  return 0;
}

