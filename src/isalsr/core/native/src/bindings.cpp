/// bindings.cpp — nanobind module definition for isalsr.core._native.
///
/// This is the single NB_MODULE entry point.  Algorithm TUs added in later
/// phases register their sub-modules here.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <isalsr/canonical.hpp>
#include <isalsr/cdll.hpp>
#include <isalsr/labeled_dag.hpp>
#include <isalsr/node_types.hpp>
#include <isalsr/string_to_dag.hpp>
#include <isalsr/wl.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

// Forward declarations from probe.cpp
std::string probe_engine_name();
std::unordered_map<std::string, std::string> probe_build_info();
uint64_t probe_fnv1a64(nb::bytes data);

NB_MODULE(_native, m) {
    m.doc() = "isalsr.core._native — C++ extension for IsalSR.";

    // -----------------------------------------------------------------------
    // CanonicalTimeoutError — raised when fast_canonical_string times out.
    // Must be registered before the functions that throw it.
    // -----------------------------------------------------------------------
    nb::exception<isalsr::CanonicalTimeoutError>(m, "CanonicalTimeoutError");

    m.def("engine_name", &probe_engine_name,
          "Return the literal string \"cpp\" identifying this engine.");

    m.def("build_info", &probe_build_info,
          "Return a dict of compiler/ISA/flag metadata recorded at build time.");

    m.def("fnv1a64", &probe_fnv1a64,
          nb::arg("data"),
          "Compute FNV-1a 64-bit hash of *data* (bytes). "
          "Returns an unsigned 64-bit integer. "
          "Offset basis 0xcbf29ce484222325, prime 0x100000001b3.");

    // -----------------------------------------------------------------------
    // testing submodule — differential-test bindings for CDLL and LabeledDAG.
    // NOT part of the production API; used only by test_native_datastructures.
    // -----------------------------------------------------------------------
    auto testing = m.def_submodule(
        "testing",
        "Internal C++ data-structure bindings for differential testing only. "
        "Not a public API.");

    // -- NativeCDLL ----------------------------------------------------------
    nb::class_<isalsr::NativeCDLL>(testing, "NativeCDLL",
        "C++ port of CircularDoublyLinkedList (cdll.py). "
        "CDLL slot indices are NOT graph node indices; payloads are graph node IDs.")
        .def(nb::init<int32_t>(), nb::arg("capacity"))
        .def("size",       &isalsr::NativeCDLL::size,
             "Return the number of active nodes.")
        .def("capacity",   &isalsr::NativeCDLL::capacity,
             "Return the maximum number of nodes.")
        .def("get_value",  &isalsr::NativeCDLL::get_value,  nb::arg("node"),
             "Return the graph node ID stored in CDLL slot `node`.")
        .def("set_value",  &isalsr::NativeCDLL::set_value,  nb::arg("node"), nb::arg("value"),
             "Overwrite the graph node ID stored in CDLL slot `node`.")
        .def("next_node",  &isalsr::NativeCDLL::next_node,  nb::arg("node"),
             "Return the successor CDLL slot index of `node`.")
        .def("prev_node",  &isalsr::NativeCDLL::prev_node,  nb::arg("node"),
             "Return the predecessor CDLL slot index of `node`.")
        .def("insert_after", &isalsr::NativeCDLL::insert_after,
             nb::arg("node"), nb::arg("value"),
             "Insert a new slot with payload `value` after slot `node`. "
             "Returns the new slot index.  When the list is empty `node` is ignored.")
        .def("remove",     &isalsr::NativeCDLL::remove,     nb::arg("node"),
             "Remove slot `node` from the list.  No-op on empty list.");

    // -- NativeLabeledDAG ----------------------------------------------------
    nb::class_<isalsr::NativeLabeledDAG>(testing, "NativeLabeledDAG",
        "C++ port of LabeledDAG (labeled_dag.py). "
        "Labels are passed/returned as integers (see NODE_TYPE_INT in the test file). "
        "var_index=-1 means absent; const_value=NaN means absent.")
        .def(nb::init<int32_t>(), nb::arg("max_nodes"))
        .def_prop_ro("node_count",
            [](const isalsr::NativeLabeledDAG& d) { return d.node_count_; })
        .def_prop_ro("edge_count",
            [](const isalsr::NativeLabeledDAG& d) { return d.edge_count_; })
        .def_prop_ro("max_nodes",
            [](const isalsr::NativeLabeledDAG& d) { return d.max_nodes_; })
        .def("add_node",
            [](isalsr::NativeLabeledDAG& dag, int label, int var_index, double const_value) {
                return dag.add_node(
                    static_cast<isalsr::NodeType>(label), var_index, const_value);
            },
            nb::arg("label"), nb::arg("var_index"), nb::arg("const_value"),
            "Add a new node.  label is an integer code (see NODE_TYPE_INT). "
            "var_index=-1 means absent.  const_value=NaN means absent.")
        .def("add_edge", &isalsr::NativeLabeledDAG::add_edge,
             nb::arg("source"), nb::arg("target"),
             "Add directed edge source→target with cycle/duplicate check. "
             "Returns True if added, False on cycle or duplicate.")
        .def("add_edge_unchecked", &isalsr::NativeLabeledDAG::add_edge_unchecked,
             nb::arg("source"), nb::arg("target"),
             "Add edge without cycle/duplicate check (acyclicity must be guaranteed).")
        .def("remove_edge", &isalsr::NativeLabeledDAG::remove_edge,
             nb::arg("source"), nb::arg("target"),
             "Remove edge source→target.  Returns True if it existed.")
        .def("undo_node", &isalsr::NativeLabeledDAG::undo_node,
             "Remove the last-added node and all its incident edges.")
        .def("has_edge", &isalsr::NativeLabeledDAG::has_edge,
             nb::arg("source"), nb::arg("target"))
        .def("out_neighbors", &isalsr::NativeLabeledDAG::out_neighbors,
             nb::arg("node"),
             "Return sorted out-neighbors of `node` as a list.")
        .def("in_neighbors", &isalsr::NativeLabeledDAG::in_neighbors,
             nb::arg("node"),
             "Return sorted in-neighbors of `node` as a list.")
        .def("ordered_inputs", &isalsr::NativeLabeledDAG::ordered_inputs,
             nb::arg("node"),
             "Return source nodes in insertion order (critical for binary ops, B9).")
        .def("node_label",
            [](const isalsr::NativeLabeledDAG& dag, int32_t node) {
                return static_cast<int>(dag.node_label(node));
            },
            nb::arg("node"),
            "Return the NodeType integer code for `node`.")
        .def("node_var_index", &isalsr::NativeLabeledDAG::node_var_index,
             nb::arg("node"),
             "Return var_index for `node`, or -1 if absent.")
        .def("node_const_value", &isalsr::NativeLabeledDAG::node_const_value,
             nb::arg("node"),
             "Return const_value for `node`, or NaN if absent.")
        .def("topological_sort", &isalsr::NativeLabeledDAG::topological_sort,
             "Return nodes in topological order (Kahn's algorithm).")
        .def("var_nodes", &isalsr::NativeLabeledDAG::var_nodes,
             "Return sorted list of VAR node IDs.")
        .def("non_var_nodes", &isalsr::NativeLabeledDAG::non_var_nodes,
             "Return sorted list of non-VAR node IDs.")
        .def("has_cycle_if_added", &isalsr::NativeLabeledDAG::has_cycle_if_added,
             nb::arg("source"), nb::arg("target"),
             "Check if adding source→target would create a cycle.")
        .def("normalize_const_creation",
             &isalsr::NativeLabeledDAG::normalize_const_creation,
             "Return a new DAG with all CONST creation edges moved to node 0.")
        .def("has_const_nodes", &isalsr::NativeLabeledDAG::has_const_nodes,
             "Return True if any CONST node exists.");

    // -----------------------------------------------------------------------
    // wl_node_hash — WL subtree FNV-1a hash (conformance oracle for tests)
    // -----------------------------------------------------------------------
    m.def("wl_node_hash",
        [](const std::string& label_value,
           const std::vector<uint64_t>& children_hashes) -> uint64_t {
            return isalsr::wl_node_hash(label_value, children_hashes);
        },
        nb::arg("label_value"), nb::arg("children_hashes"),
        "Compute 1-WL FNV-1a 64-bit node hash.\n\n"
        "``label_value`` is the NodeType value string (e.g. '+' for ADD, 'var' for VAR).\n"
        "``children_hashes`` must already be sorted (caller responsibility).\n"
        "Matches Python's _wl_node_hash in canonical.py byte-for-byte.");

    // -----------------------------------------------------------------------
    // fast_canonical_string — main production entry point
    // -----------------------------------------------------------------------
    m.def("fast_canonical_string",
        [](const isalsr::NativeLabeledDAG& dag, double timeout_sec) -> std::string {
            return isalsr::fast_canonical_string(dag, timeout_sec);
        },
        nb::arg("dag"), nb::arg("timeout_sec"),
        "Compute fast canonical string (wl_only mode) for a NativeLabeledDAG.\n\n"
        "Applies normalize_const_creation() internally when CONST nodes are present.\n"
        "``timeout_sec`` < 0 means no limit.\n"
        "Raises CanonicalTimeoutError if the budget is exceeded.");

    // -- tokenize ------------------------------------------------------------
    testing.def("tokenize",
        [](const std::string& s) { return isalsr::tokenize(s); },
        nb::arg("input_string"),
        "Tokenize an IsalSR instruction string into a list of tokens (str). "
        "Raises ValueError on invalid characters or truncated V/v.");

    // -- NativeStringToDAG ---------------------------------------------------
    nb::class_<isalsr::NativeStringToDAG>(testing, "NativeStringToDAG",
        "C++ port of StringToDAG (string_to_dag.py). "
        "Tokenizes and executes an IsalSR instruction string to produce a NativeLabeledDAG.")
        .def(nb::init<const std::string&, int32_t>(),
             nb::arg("input_string"), nb::arg("num_variables"),
             "Construct and tokenize.  Raises ValueError on invalid tokens or num_variables < 1.")
        .def("run", &isalsr::NativeStringToDAG::run,
             "Execute the string-to-DAG conversion. "
             "Returns an independent NativeLabeledDAG (by value).")
        .def("tokens", &isalsr::NativeStringToDAG::tokens,
             "Return the tokenized instruction list (single- or two-char strings).");
}
