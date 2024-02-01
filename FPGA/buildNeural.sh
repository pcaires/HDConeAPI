#!/bin/bash
export PATH=/glob/intel-python/python3/bin/:/glob/intel-python/python2/bin/:${PATH}
source /glob/development-tools/versions/oneapi/2022.3.1/oneapi/setvars.sh --force
echo $QUARTUS_ROOTDIR_OVERRIDE
export QUARTUS_ROOTDIR_OVERRIDE=/glob/development-tools/versions/oneapi/2022.3.1/intelFPGA_pro/19.2/quartus
echo $QUARTUS_ROOTDIR_OVERRIDE
echo "inference fixed"
dpcpp -fintelfpga src/hdcneuralhd.cpp -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10 -o ./bin/neural.fpga -DFPGA=1 -O2 -fp-model=fast=2 -Xsffp-reassociate
 -qactypes -no-fma -fp-model=precise -I $INTELFPGAOCLSDKROOT/include/ref -qactypes -Xsparallel=12 -Wno-invalid-constexpr
# dpcpp -fintelfpga src/hdc21-inferencePipesRealBFloat.cpp -Xshardware -Xsprofile -Xsboard=intel_s10sx_pac:pac_s10 -o ./bin/codefromhdc21inferencePipes16BFloat256.fpga -no-fma -fp-model=precise -I $INTELFPGAOCLSDKROOT/include/ref -qactypes -DFPGA=1 -O2
#-Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10

# #!/bin/bash
# source /opt/intel/inteloneapi/setvars.sh
# make hw