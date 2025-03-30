/* 
 * XIBRA Network Secure Enclave Implementation
 * Intel SGX Protected Core for Confidential AI Operations
 */

#include <sgx_urts.h>
#include <sgx_tcrypto.h>
#include <sgx_tseal.h>
#include <sgx_utils.h>
#include "Enclave_impl_u.h"

#define ENCLAVE_FILE "xibra_enclave.signed.so"
#define MAX_BUFFER_LEN 4096
#define KEY_DERIVATION_CONTEXT "XIBRA-KEY-1.0"

/* Secure Enclave Global Context */
typedef struct {
    sgx_enclave_id_t eid;
    sgx_ra_context_t ra_ctx;
    sgx_ec256_public_t sp_pubkey;
} enclave_ctx_t;

/* Enclave Initialization */
sgx_status_t initialize_enclave(enclave_ctx_t *ctx) {
    sgx_launch_token_t token = {0};
    int updated = 0;
    
    // 1. Create Enclave
    sgx_status_t ret = sgx_create_enclave(ENCLAVE_FILE, SGX_DEBUG_FLAG, 
                                        &token, &updated, 
                                        &ctx->eid, NULL);
    if (ret != SGX_SUCCESS) {
        return ret;
    }

    // 2. Initialize Remote Attestation
    ret = sgx_ra_init(&ctx->sp_pubkey, 0, &ctx->ra_ctx, ctx->eid);
    if (ret != SGX_SUCCESS) {
        sgx_destroy_enclave(ctx->eid);
        return ret;
    }

    return SGX_SUCCESS;
}

/* Secure Key Derivation */
sgx_status_t derive_secure_key(enclave_ctx_t *ctx, 
                             const uint8_t *salt, size_t salt_len,
                             uint8_t *out_key, size_t key_len) {
    sgx_status_t ret;
    sgx_sha_state_handle_t sha_handle;
    sgx_sha256_hash_t hash;
    
    // 1. Start HMAC-SHA256 Chain
    if ((ret = sgx_sha256_init(&sha_handle)) != SGX_SUCCESS) {
        return ret;
    }

    // 2. Update with Context String
    if ((ret = sgx_sha256_update((const uint8_t*)KEY_DERIVATION_CONTEXT, 
                               strlen(KEY_DERIVATION_CONTEXT), 
                               sha_handle)) != SGX_SUCCESS) {
        sgx_sha256_close(sha_handle);
        return ret;
    }

    // 3. Update with Salt
    if ((ret = sgx_sha256_update(salt, salt_len, sha_handle)) != SGX_SUCCESS) {
        sgx_sha256_close(sha_handle);
        return ret;
    }

    // 4. Finalize Hash
    if ((ret = sgx_sha256_get_hash(sha_handle, &hash)) != SGX_SUCCESS) {
        sgx_sha256_close(sha_handle);
        return ret;
    }
    sgx_sha256_close(sha_handle);

    // 5. HKDF Expansion
    return ecall_key_derivation(ctx->eid, &ret, 
                              hash, sizeof(hash),
                              out_key, key_len);
}

/* Data Sealing API */
sgx_status_t seal_sensitive_data(enclave_ctx_t *ctx,
                               const uint8_t *data, size_t data_len,
                               uint8_t *sealed, size_t sealed_len) {
    sgx_status_t ret;
    uint32_t sealed_size;
    
    // 1. Get Required Buffer Size
    if ((ret = sgx_calc_sealed_data_size(0, data_len)) != SGX_SUCCESS) {
        return ret;
    }
    sealed_size = ret;
    
    if (sealed_len < sealed_size) {
        return SGX_ERROR_INVALID_PARAMETER;
    }

    // 2. Perform Sealing Inside Enclave
    return ecall_seal_data(ctx->eid, &ret, 
                          data, data_len,
                          sealed, sealed_size);
}

/* Remote Attestation Handshake */
sgx_status_t generate_attestation_evidence(enclave_ctx_t *ctx,
                                          sgx_ra_msg1_t *msg1) {
    sgx_status_t ret;
    return sgx_ra_get_msg1(ctx->ra_ctx, ctx->eid, 
                          sgx_ra_get_ga, msg1);
}

sgx_status_t verify_attestation_response(enclave_ctx_t *ctx,
                                       const sgx_ra_msg2_t *msg2,
                                       const sgx_ra_msg3_t **msg3,
                                       size_t msg3_size) {
    return sgx_ra_proc_msg2(ctx->ra_ctx, ctx->eid, 
                           sgx_ra_proc_msg2_trusted,
                           sgx_ra_get_msg3_trusted,
                           (sgx_ra_msg2_t*)msg2, 
                           msg3_size, 
                           (sgx_ra_msg3_t**)msg3);
}

/* Secure Memory Management */
void secure_memzero(void *ptr, size_t len) {
    volatile uint8_t *p = (volatile uint8_t*)ptr;
    while (len--) {
        *p++ = 0;
    }
    __asm__ __volatile__ ("" : : "r"(ptr) : "memory");
}

/* Enclave Teardown */
void destroy_enclave(enclave_ctx_t *ctx) {
    if (ctx->eid) {
        sgx_ra_close(ctx->ra_ctx, ctx->eid);
        sgx_destroy_enclave(ctx->eid);
        memset_s(ctx, sizeof(enclave_ctx_t), 0, sizeof(enclave_ctx_t));
    }
}
