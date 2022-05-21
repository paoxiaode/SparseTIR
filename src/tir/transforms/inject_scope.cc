/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file inject_scope.cc
 */
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/analyzer.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {

class VarCollector : public StmtExprVisitor {
 public:
  std::unordered_set<const VarNode*> used;

 private:
  void VisitExpr_(const VarNode* op) final { used.insert(op); }
};

}  // namespace

class ScopeInjector : public StmtExprMutator {
 public:
  explicit ScopeInjector() : loop_stack_(1) {}

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_stack_.back().push_back(GetRef<For>(op));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    loop_stack_.push_back({});
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    loop_stack_.pop_back();
    return ret;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    PrimExpr cond = const_true();
    arith::Analyzer ana;
    VarCollector collector;
    collector(GetRef<BlockRealize>(op));
    Stmt body = StmtExprMutator::VisitStmt_(op);
    if (op->block->annotations.count("atomic")) {
      for (const For& loop : loop_stack_.back()) {
        if (!collector.used.count(loop->loop_var.get())) {
          cond = cond && (loop->loop_var == loop->min);
        }
      }
      return IfThenElse(ana.Simplify(cond), body);
    } else {
      return body;
    }
  }

  std::vector<Array<For>> loop_stack_;
};

PrimFunc InjectScope(PrimFunc f) {
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    ScopeInjector injector;
    fptr->body = injector(fptr->body);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass InjectScope() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return InjectScope(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectScope", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectScope").set_body_typed(InjectScope);

}  // namespace transform

}  // namespace tir
}  // namespace tvm