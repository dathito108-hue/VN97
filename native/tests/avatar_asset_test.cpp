#include "vn97/avatar_asset.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr std::size_t kHeader = 80;
constexpr std::size_t kVertexStride = 32;
constexpr std::size_t kJointStride = 48;

void W16(std::uint8_t* p, std::uint16_t v) { p[0]=v&255; p[1]=(v>>8)&255; }
void W32(std::uint8_t* p, std::uint32_t v) { for(int i=0;i<4;++i)p[i]=(v>>(8*i))&255; }
void W64(std::uint8_t* p, std::uint64_t v) { for(int i=0;i<8;++i)p[i]=(v>>(8*i))&255; }
void WI32(std::uint8_t* p, std::int32_t v){ std::uint32_t b=0; std::memcpy(&b,&v,4); W32(p,b); }
void WF32(std::uint8_t* p, float v){ std::uint32_t b=0; std::memcpy(&b,&v,4); W32(p,b); }
std::uint32_t Crc(const std::uint8_t* data,std::size_t n){ std::uint32_t c=0xffffffffu; for(std::size_t i=0;i<n;++i){ c^=data[i]; for(int b=0;b<8;++b){ auto m=0u-(c&1u); c=(c>>1)^(0xedb88320u&m);} } return ~c; }

std::vector<std::uint8_t> Build() {
    const std::uint32_t vc=3, ic=3, jc=2;
    const std::size_t vo=kHeader;
    const std::size_t io=vo+vc*kVertexStride;
    const std::size_t jo=io+ic*4;
    const std::size_t total=jo+jc*kJointStride;
    std::vector<std::uint8_t> b(total,0);
    const char magic[8]={'V','N','9','7','A','V','1','\0'};
    std::memcpy(b.data(),magic,8); W16(b.data()+8,1); W16(b.data()+10,80);
    W32(b.data()+16,vc); W32(b.data()+20,ic); W32(b.data()+24,jc);
    W32(b.data()+28,32); W32(b.data()+32,48); W64(b.data()+40,vo); W64(b.data()+48,io); W64(b.data()+56,jo); W64(b.data()+64,total);
    const float pos[3][3]={{-1,0,0},{1,0,0},{0,1,0}};
    for(std::size_t i=0;i<vc;++i){ auto* p=b.data()+vo+i*32; for(int c=0;c<3;++c)WF32(p+c*4,pos[i][c]); WF32(p+12,0);WF32(p+16,0);WF32(p+20,1); p[24]=0;p[25]=1;p[28]=200;p[29]=55; }
    W32(b.data()+io,0); W32(b.data()+io+4,1); W32(b.data()+io+8,2);
    auto* j0=b.data()+jo; WI32(j0,-1); WF32(j0+4,0);WF32(j0+8,0);WF32(j0+12,0); WF32(j0+16,0);WF32(j0+20,0);WF32(j0+24,0);WF32(j0+28,1); WF32(j0+32,1);WF32(j0+36,1);WF32(j0+40,1);
    auto* j1=b.data()+jo+48; WI32(j1,0); WF32(j1+4,0);WF32(j1+8,1);WF32(j1+12,0); WF32(j1+16,0);WF32(j1+20,0);WF32(j1+24,0);WF32(j1+28,1); WF32(j1+32,1);WF32(j1+36,1);WF32(j1+40,1);
    W32(b.data()+72,Crc(b.data()+80,b.size()-80)); W32(b.data()+76,Crc(b.data(),76)); return b;
}
void Refresh(std::vector<std::uint8_t>& b){ W32(b.data()+72,Crc(b.data()+80,b.size()-80)); W32(b.data()+76,Crc(b.data(),76)); }
}

int main(){
    auto blob=Build(); vn97::AvatarAssetView view;
    assert(vn97::ParseAvatarAsset(blob.data(),blob.size(),&view)==vn97::AvatarAssetStatus::kOk);
    assert(view.vertex_count==3 && view.index_count==3 && view.joint_count==2);
    vn97::AvatarVertex v; assert(vn97::ReadAvatarVertex(view,2,&v)==vn97::AvatarAssetStatus::kOk); assert(std::fabs(v.position[1]-1.0f)<1e-6f); assert(v.joint_weights[0]+v.joint_weights[1]==255);
    std::uint32_t idx=99; assert(vn97::ReadAvatarIndex(view,1,&idx)==vn97::AvatarAssetStatus::kOk && idx==1);
    vn97::AvatarJoint j; assert(vn97::ReadAvatarJoint(view,1,&j)==vn97::AvatarAssetStatus::kOk && j.parent==0);
    std::uint32_t vc=0,ic=0,jc=0; assert(vn97_avatar_asset_parse(blob.data(),blob.size(),&vc,&ic,&jc)==0 && vc==3 && ic==3 && jc==2);

    auto bad=blob; bad[80]^=1; assert(vn97::ParseAvatarAsset(bad.data(),bad.size(),&view)==vn97::AvatarAssetStatus::kChecksumMismatch);
    bad=blob; W32(bad.data()+kHeader+3*kVertexStride+8,3); Refresh(bad); assert(vn97::ParseAvatarAsset(bad.data(),bad.size(),&view)==vn97::AvatarAssetStatus::kIndexOutOfRange);
    bad=blob; bad[80+28]=254; bad[80+29]=0; Refresh(bad); assert(vn97::ParseAvatarAsset(bad.data(),bad.size(),&view)==vn97::AvatarAssetStatus::kInvalidRig);
    bad=blob; WI32(bad.data()+80+3*32+3*4+48,1); Refresh(bad); assert(vn97::ParseAvatarAsset(bad.data(),bad.size(),&view)==vn97::AvatarAssetStatus::kInvalidRig);
    bad=blob; WF32(bad.data()+80,std::numeric_limits<float>::quiet_NaN()); Refresh(bad); assert(vn97::ParseAvatarAsset(bad.data(),bad.size(),&view)==vn97::AvatarAssetStatus::kNonFinite);
    bad=blob; W32(bad.data()+28,31); W32(bad.data()+76,Crc(bad.data(),76)); assert(vn97::ParseAvatarAsset(bad.data(),bad.size(),&view)==vn97::AvatarAssetStatus::kInvalidHeader);
    return 0;
}
