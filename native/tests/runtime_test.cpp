#include "vn97/runtime.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
int main(){
  vn97_runtime_config cfg{2,1,3,2,0,0}; std::uint64_t h=0;
  assert(vn97_runtime_create(&cfg,&h)==0 && h!=0);
  vn97_runtime_info info{}; assert(vn97_runtime_info_get(h,&info)==0); assert(info.state_count==12); assert(info.lifecycle==0); assert(info.resolved_recurrent_backend==1);
  std::vector<float> st(12); for(size_t i=0;i<st.size();++i) st[i]=float(i)+0.25f;
  assert(vn97_runtime_state_write(h,st.data(),st.size())==0);
  assert(vn97_runtime_state_write(h,st.data(),st.size()-1)==8);
  auto nonfinite=st; nonfinite[0]=std::numeric_limits<float>::infinity();
  assert(vn97_runtime_state_write(h,nonfinite.data(),nonfinite.size())==2);
  assert(vn97_runtime_checkpoint_size(h,nullptr)==1);
  assert(vn97_runtime_activate(h)==0); assert(vn97_runtime_state_write(h,st.data(),st.size())==5); assert(vn97_runtime_advance(h,7)==0); assert(vn97_runtime_suspend(h)==0);
  size_t size=0; assert(vn97_runtime_checkpoint_size(h,&size)==0); std::vector<std::uint8_t> blob(size); size_t written=0; assert(vn97_runtime_checkpoint_write(h,blob.data(),blob.size()-1,&written)==6 && written==0); assert(vn97_runtime_checkpoint_write(h,blob.data(),blob.size(),&written)==0 && written==size);
  assert(vn97_runtime_destroy(h)==0); assert(vn97_runtime_info_get(h,&info)==4);
  std::uint64_t r=0; assert(vn97_runtime_restore(blob.data(),blob.size(),&r)==0); assert(vn97_runtime_info_get(r,&info)==0); assert(info.lifecycle==2 && info.sequence_position==7 && info.state_count==12);
  std::vector<float> restored(12); assert(vn97_runtime_state_read(r,restored.data(),restored.size())==0); assert(restored==st);
  auto bad=blob; bad.back()^=1; std::uint64_t badh=0; assert(vn97_runtime_restore(bad.data(),bad.size(),&badh)==7);
  auto bad_header=blob; bad_header[12]^=1; assert(vn97_runtime_restore(bad_header.data(),bad_header.size(),&badh)==7);
  assert(vn97_runtime_resume(r)==0); assert(vn97_runtime_advance(r,2)==0); assert(vn97_runtime_suspend(r)==0); assert(vn97_runtime_destroy(r)==0);
  vn97_runtime_config neon{1,1,1,1,2,1}; std::uint64_t n=0; assert(vn97_runtime_create(&neon,&n)==10);
  vn97_runtime_config huge{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu,1,1}; assert(vn97_runtime_create(&huge,&n)==3);
  return 0;
}
