#include <stdio.h>
#include <stdlib.h>
#include <infiniband/verbs.h>

int main() {
    struct ibv_context **ctx_list;
    struct ibv_device **dev_list;
    struct ibv_device_attr_ex dev_attr_ex;
    int num_devices, i;

    // Get the list of RDMA devices
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        perror("Failed to get RDMA device list");
        return 1;
    }

    if (num_devices == 0) {
        printf("No RDMA devices found.\n");
        ibv_free_device_list(dev_list);
        return 1;
    }

    printf("Found %d RDMA devices.\n", num_devices);

    for (i = 0; i < num_devices; i++) {
        struct ibv_context *ctx = ibv_open_device(dev_list[i]);
        if (!ctx) {
            perror("Failed to open RDMA device");
            continue;
        }

        // Query device extended attributes
        if (ibv_query_device_ex(ctx, NULL, &dev_attr_ex) != 0) {
            perror("Failed to query device extended attributes");
            ibv_close_device(ctx);
            continue;
        }

        printf("Device: %s\n", ibv_get_device_name(dev_list[i]));

        // Check ODP support in general
        if (dev_attr_ex.odp_caps.general_caps & IBV_ODP_SUPPORT) {
            printf("  ODP is supported.\n");

            // Check specific ODP capabilities
            if (dev_attr_ex.odp_caps.per_transport_caps.rc_odp_caps & IBV_ODP_SUPPORT_WRITE)
                printf("  - RC ODP Write is supported.\n");
            if (dev_attr_ex.odp_caps.per_transport_caps.rc_odp_caps & IBV_ODP_SUPPORT_READ)
                printf("  - RC ODP Read is supported.\n");
            if (dev_attr_ex.odp_caps.per_transport_caps.ud_odp_caps & IBV_ODP_SUPPORT_WRITE)
                printf("  - UD ODP Write is supported.\n");
            if (dev_attr_ex.odp_caps.per_transport_caps.ud_odp_caps & IBV_ODP_SUPPORT_READ)
                printf("  - UD ODP Read is supported.\n");
        } else {
            printf("  ODP is not supported on this device.\n");
        }

        // Close the device context
        ibv_close_device(ctx);
    }

    // Free the device list
    ibv_free_device_list(dev_list);

    return 0;
}
