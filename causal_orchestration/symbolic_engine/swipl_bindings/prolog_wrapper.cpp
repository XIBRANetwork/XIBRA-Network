// XIBRA Prolog Engine Wrapper
// SWI-Prolog C++ Interface with RAII Management

#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <swipl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace xibra::prolog {

class Engine;
class Query;

// Exception hierarchy
class PrologException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InitializationError : public PrologException {
public:
    InitializationError(const char* msg) : PrologException(msg) {}
};

// Thread-safe Prolog engine manager
class EnginePool {
public:
    EnginePool(size_t pool_size = 4, const char* init_argv[] = nullptr) {
        std::call_once(init_flag_, [&]() {
            if (!PL_initialise(0, nullptr)) {
                throw InitializationError("Prolog engine initialization failed");
            }
        });
        
        for(size_t i = 0; i < pool_size; ++i) {
            engines_.emplace_back(std::make_unique<Engine>());
        }
    }

    Engine& acquire_engine() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this]{ return !engines_.empty(); });
        
        auto& engine = engines_.back();
        engines_.pop_back();
        return *engine;
    }

    void release_engine(Engine& engine) {
        std::lock_guard lock(mutex_);
        engines_.push_back(std::unique_ptr<Engine>(&engine));
        cv_.notify_one();
    }

private:
    std::vector<std::unique_ptr<Engine>> engines_;
    std::mutex mutex_;
    std::condition_variable cv_;
    static std::once_flag init_flag_;
};

std::once_flag EnginePool::init_flag_;

// RAII-style Prolog engine wrapper
class Engine {
public:
    Engine() {
        if (!PL_thread_attach_engine(nullptr)) {
            throw InitializationError("Failed to attach Prolog engine");
        }
    }

    ~Engine() {
        PL_thread_destroy_engine();
    }

    Query create_query(const std::string& predicate);

    // Data conversion utilities
    static json term_to_json(term_t t);
    static term_t json_to_term(const json& j);

private:
    friend class Query;
    std::mutex engine_mutex_;
};

// Prolog query execution context
class Query {
public:
    Query(Engine& engine, const std::string& predicate)
        : engine_(engine), predicate_(predicate) 
    {
        qid_ = PL_open_query(nullptr, PL_Q_NORMAL, 
            PL_new_term_refs(0), predicate.c_str());
        if (qid_ == 0) {
            throw PrologException("Query creation failed");
        }
    }

    ~Query() {
        PL_close_query(qid_);
    }

    bool next_solution() {
        std::lock_guard lock(engine_.engine_mutex_);
        return PL_next_solution(qid_);
    }

    template<typename T>
    void bind_argument(size_t pos, const T& value);

    json get_result() const {
        return Engine::term_to_json(PL_new_term_ref());
    }

private:
    Engine& engine_;
    qid_t qid_;
    std::string predicate_;
};

// Data conversion implementation
json Engine::term_to_json(term_t t) {
    json result;
    
    if (PL_is_variable(t)) {
        result = nullptr;
    } else if (PL_is_atom(t)) {
        char* s;
        PL_get_atom_chars(t, &s);
        result = std::string(s);
    } else if (PL_is_integer(t)) {
        long l;
        PL_get_long(t, &l);
        result = l;
    } else if (PL_is_compound(t)) {
        functor_t functor = PL_new_functor(PL_new_atom("json"), 1);
        if (PL_is_functor(t, functor)) {
            term_t arg = PL_new_term_ref();
            PL_get_arg(1, t, arg);
            return term_to_json(arg);
        }
        
        json arr;
        term_t head = PL_new_term_ref();
        term_t tail = PL_new_term_ref();
        PL_get_list(t, head, tail);
        
        while(PL_is_list(t)) {
            arr.push_back(term_to_json(head));
            PL_get_list(t, head, tail);
            t = tail;
        }
        result = arr;
    }
    
    return result;
}

term_t Engine::json_to_term(const json& j) {
    term_t t = PL_new_term_ref();
    
    if (j.is_null()) {
        PL_put_variable(t);
    } else if (j.is_string()) {
        PL_put_atom_chars(t, j.get<std::string>().c_str());
    } else if (j.is_number_integer()) {
        PL_put_long(t, j.get<long>());
    } else if (j.is_array()) {
        term_t list = PL_new_term_ref();
        PL_put_nil(list);
        
        for(auto it = j.rbegin(); it != j.rend(); ++it) {
            term_t head = json_to_term(*it);
            PL_cons_list(list, head, list);
        }
        PL_unify(t, list);
    }
    
    return t;
}

// Template specialization for argument binding
template<>
void Query::bind_argument<json>(size_t pos, const json& value) {
    term_t arg = Engine::json_to_term(value);
    PL_unify_arg(pos, PL_term_ref(qid_), arg);
}

} // namespace xibra::prolog

// Example usage
int main() {
    using namespace xibra::prolog;
    
    try {
        EnginePool pool(4);
        auto& engine = pool.acquire_engine();
        
        Query query(engine, "xibra_rules:validate_operation(Agent, Action, Resource, Context)");
        query.bind_argument<json>(1, {{"type", "ai_agent"}, {"id", 42}});
        
        while(query.next_solution()) {
            auto result = query.get_result();
            std::cout << "Validation result: " << result.dump() << std::endl;
        }
        
        pool.release_engine(engine);
    } catch(const PrologException& e) {
        std::cerr << "Prolog error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
