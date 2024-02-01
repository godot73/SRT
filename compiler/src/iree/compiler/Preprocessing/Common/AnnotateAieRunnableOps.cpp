#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

namespace {

// TODO(sungsoon)
struct AnnotateAieRunnableOpsPass
    : AnnotateAieRunnableOpsBase<AnnotateAieRunnableOpsPass> {

  AnnotateAieRunnableOpsPass() {
    fprintf(stderr, "yyyyyyyy AnnotateAieRunnableOpsPass ctor\n");
  }

  void runOnOperation() override {
    fprintf(stderr, "yyyyyyyy AnnotateAieRunnableOpsPass::runOnOperation()\n");
  }
};

} // namespace

std::unique_ptr<Pass> createAnnotateAieRunnableOpsPass() {
  return std::make_unique<AnnotateAieRunnableOpsPass>();
}

} // namespace mlir::iree_compiler::Preprocessing

