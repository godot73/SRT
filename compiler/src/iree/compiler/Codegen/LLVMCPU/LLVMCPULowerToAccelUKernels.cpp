// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
class LLVMCPULowerToAccelUKernelsPass
    : public LLVMCPULowerToAccelUKernelsBase<LLVMCPULowerToAccelUKernelsPass> {
public:
  LLVMCPULowerToAccelUKernelsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LLVMCPULowerToAccelUKernelsPass::runOnOperation() {
  // TODO
}

std::unique_ptr<OperationPass<>> createLLVMCPULowerToAccelUKernelsPass() {
  return std::make_unique<LLVMCPULowerToAccelUKernelsPass>();
}

} // namespace iree_compiler
} // namespace mlir
