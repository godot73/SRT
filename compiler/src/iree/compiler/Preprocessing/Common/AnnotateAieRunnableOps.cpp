#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

std::unique_ptr<Pass> createAnnotateAieRunnableOpsPass() {
  // TODO(sungsoon)
  return nullptr;
}

} // namespace mlir::iree_compiler::Preprocessing

