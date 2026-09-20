// Does an arm's number depend on its POSITION in the harness's fixed arm order?
// At the largest working set the three input payloads plus the output exceed L3,
// so whichever arm runs first leaves its input resident and the next arm finds it
// evicted. Three schedules, same work:
//   fwd    seq then intlv          (what the harness does)
//   rev    intlv then seq
//   scrub  each arm preceded by a full-L3 scrub, so neither inherits residency
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <functional>
#include <arrow/util/bpacking_internal.h>
#include "arrow/util/fastlanes/interleaved_pfor.h"
#include "corpus_generators.h"

// Everything below is private to this file. One study per translation unit,
// so a study's kernels are compiled exactly as they were when it was a
// standalone binary, and two studies can hold the same name for different
// things.
namespace {

namespace fl = arrow::util::fastlanes;
using fl::InterleavedPforOrder;
using Clock = std::chrono::steady_clock;
static constexpr size_t kBlk = 1024;

struct Seq { std::vector<uint8_t> bytes; std::vector<uint32_t> off;
             std::vector<uint8_t> w; std::vector<int32_t> mn; };
static Seq EncSeq(const std::vector<int32_t>& v) {
  size_t nb = v.size()/kBlk; Seq p; p.w.resize(nb); p.mn.resize(nb); p.off.resize(nb);
  size_t o=0;
  for (size_t b=0;b<nb;++b){ int32_t mn=v[b*kBlk],mx=mn;
    for(size_t t=1;t<kBlk;++t){mn=std::min(mn,v[b*kBlk+t]);mx=std::max(mx,v[b*kBlk+t]);}
    uint32_t span=(uint32_t)mx-(uint32_t)mn,w=0; while(w<32&&(span>>w)!=0)++w;
    p.w[b]=w; p.mn[b]=mn; p.off[b]=o; o+=w*kBlk/8; }
  p.bytes.assign(o+64,0);
  for(size_t b=0;b<nb;++b){ uint32_t w=p.w[b]; if(!w) continue;
    uint8_t* out=p.bytes.data()+p.off[b]; uint64_t acc=0; int bits=0;
    for(size_t t=0;t<kBlk;++t){ uint32_t r=(uint32_t)v[b*kBlk+t]-(uint32_t)p.mn[b];
      acc|=(uint64_t)r<<bits; bits+=(int)w;
      while(bits>=8){*out++=(uint8_t)acc; acc>>=8; bits-=8;} }
    if(bits)*out=(uint8_t)acc; }
  return p;
}
}  // namespace


int RunLaneOrderControl(int argc, char** argv) {
  const size_t n = (argc>1)?strtoull(argv[1],nullptr,10):8192ull*1024;
  auto v = corpus::delta_shapes::GenEventMillis<int32_t>((int64_t)n);
  Seq sp = EncSeq(v);
  std::vector<uint8_t> ib(fl::InterleavedPforMaxEncodedSize(n));
  fl::InterleavedPforEncode<InterleavedPforOrder::kFileOrder>(v.data(), n, ib.data());
  void* raw=nullptr; if(posix_memalign(&raw,4096,n*4+4096)) abort();
  int32_t* out=(int32_t*)raw; memset(out,0,n*4);

  // 64 MiB scrub buffer -- larger than this host's 36 MiB L3.
  const size_t kScrub = 64ull<<20;
  std::vector<uint8_t> scrub(kScrub, 1);
  volatile uint64_t sink = 0;
  auto flush = [&]{ uint64_t a=0; for(size_t i=0;i<kScrub;i+=64) a+=scrub[i]; sink=a; };

  arrow::internal::UnpackOptions o; o.batch_size=(int64_t)kBlk;
  auto a_seq=[&]{ size_t nb=n/kBlk;
    for(size_t b=0;b<nb;++b){ uint32_t w=sp.w[b]; uint32_t* d=(uint32_t*)out+b*kBlk;
      uint32_t bias=(uint32_t)sp.mn[b];
      if(!w){ for(size_t t=0;t<kBlk;++t)d[t]=bias; continue; }
      o.bit_width=(int)w; o.max_read_bytes=(int64_t)w*128;
      arrow::internal::unpack_bias<uint32_t>(sp.bytes.data()+sp.off[b],d,o,bias);} };
  auto a_int=[&]{ fl::InterleavedPforDecode<InterleavedPforOrder::kFileOrder>(ib.data(),n,out); };

  const size_t iters = std::max<size_t>(8, (1ull<<30)/(n*4));
  auto time1=[&](const std::function<void()>& f, bool do_flush){
    double best=0;
    for(int r=0;r<7;++r){ if(do_flush) flush(); else f();
      auto t0=Clock::now(); for(size_t i=0;i<iters;++i) f();
      double s=std::chrono::duration<double>(Clock::now()-t0).count();
      best=std::max(best,(double)iters*n*4/s/(1024.0*1024*1024)); }
    return best; };

  printf("EventMillis, n=%zu (%zu MiB out, %zu MiB seq in, %zu MiB intlv in), L3=36 MiB\n\n",
         n, n*4>>20, sp.bytes.size()>>20, ib.size()>>20);
  printf("%-34s %8s %8s %10s\n","schedule","seq","intlv","intlv/seq");
  double s1=time1(a_seq,false), i1=time1(a_int,false);
  printf("%-34s %8.1f %8.1f %9.3fx\n","fwd: seq first, then intlv",s1,i1,i1/s1);
  double i2=time1(a_int,false), s2=time1(a_seq,false);
  printf("%-34s %8.1f %8.1f %9.3fx\n","rev: intlv first, then seq",s2,i2,i2/s2);
  double s3=time1(a_seq,true), i3=time1(a_int,true);
  printf("%-34s %8.1f %8.1f %9.3fx\n","scrub L3 before every arm",s3,i3,i3/s3);
  return 0;
}
