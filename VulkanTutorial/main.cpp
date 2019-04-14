#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <iostream>
#include <cstdio>

#include <vector>
#include <array>
#include <set>
#include <optional>
#include <algorithm>
#include <functional>
#include <iterator>
#include <fstream>

template <typename T>
static std::optional<uint32_t> get_index_if(std::vector<T> data, std::function<bool(const T&, const uint32_t&)> pred)
{
    uint32_t i = 0;
    for (const auto& elem : data)
    {
        if (pred(elem, i))
        {
            return std::optional<uint32_t>(i);
        }
        i++;
    }

    return std::optional<uint32_t>();
}

static std::vector<char> read_file(const std::string& filename);

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
};

QueueFamilyIndices get_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface);

struct SwapChainSupport
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct VulkanContext
{
    // vulkan instance
    VkInstance vulkan;
    VkDebugUtilsMessengerEXT debugMessenger;

    // window surface
    VkSurfaceKHR surface;

    // device and queues
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    // swap chain
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    // render pipeline
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    // command pools/buffers
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    // semaphores/fences
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    // Buffers
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
};

SwapChainSupport query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface);
static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

const uint32_t k_invalidMemoryType = static_cast<uint32_t>(-1);
uint32_t find_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

struct Vertex
{
    glm::vec2 position;
    glm::vec3 color;
    
    static VkVertexInputBindingDescription get_binding_description()
    {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::vector<VkVertexInputAttributeDescription> get_attribute_descriptions()
    {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions;

        attributeDescriptions.resize(2);

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

int main()
{
    GLFWwindow* window = nullptr;
    const uint32_t k_screenWidth = 1280, k_screenHeight = 720;

    std::vector<const char*> enabledExtensions;

    const uint32_t k_validationLayerCount = 1 ;
    const char* validationLayers[k_validationLayerCount]{
        "VK_LAYER_LUNARG_standard_validation",
    };

    const bool k_enableValidation = true;
    uint32_t enabledValidationLayers = (k_enableValidation) ? k_validationLayerCount : 0;

    struct App
    {
        bool frameBufferResized = false;
        VulkanContext* vk;
        GLFWwindow* window;
    };
    App app;

    // Init GLFW and window
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(k_screenWidth, k_screenHeight, "VulkanTutorial", nullptr, nullptr);
        app.window = window;

        glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            switch (action)
            {
            case GLFW_PRESS: if (key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(window, GLFW_TRUE); } break;
            default: break;
            }
        });
        glfwSetWindowUserPointer(window, &app);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height)
        { 
            auto app = reinterpret_cast<App*>(glfwGetWindowUserPointer(window));
            app->frameBufferResized = true;
        });

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        enabledExtensions.insert(enabledExtensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}},
    };

    const std::vector<uint16_t> indices = {
        0, 1, 2,
        2, 3, 0,
    };

    // Init Vulkan
    VulkanContext vk;
    app.vk = &vk;
    {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Tutorial";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
        createInfo.enabledLayerCount = enabledValidationLayers;
        createInfo.ppEnabledLayerNames = validationLayers;

        if (vkCreateInstance(&createInfo, nullptr, &vk.vulkan) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan instance.");
        }
    }

    // Setup debug messenger
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = vk_debug;
        createInfo.pUserData = nullptr;

        auto vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vk.vulkan, "vkCreateDebugUtilsMessengerEXT");

        if (vkCreateDebugUtilsMessengerEXT != nullptr)
        {
            vkCreateDebugUtilsMessengerEXT(vk.vulkan, &createInfo, nullptr, &vk.debugMessenger);
        }
        else
        {
            throw std::runtime_error("Failed to find vkCreateDebugUtilsMessengerEXT");
        }
    }

    // Create window surface
    {
        if (glfwCreateWindowSurface(vk.vulkan, window, nullptr, &vk.surface) != VK_SUCCESS)
        {
            throw new std::runtime_error("Failed to create window surface.");
        }
    }

    // Find physical device
    {
        // get all physical devices
        std::uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(vk.vulkan, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(vk.vulkan, &deviceCount, devices.data());

        std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        // select physical device
        for (const auto& device : devices)
        {
            VkPhysicalDeviceProperties deviceProperties;
            VkPhysicalDeviceFeatures deviceFeatures;

            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

            // put property/feature calculations here.
            bool deviceHasNecessaryPropertiesAndFeatures = true;

            // find queues
            

            // check for necessary extensions
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
            
            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

            for (const auto& extension : availableExtensions)
            {
                requiredExtensions.erase(extension.extensionName);
            }

            bool deviceHasNecessaryExtensions = requiredExtensions.empty();

            bool deviceHasSufficientSwapChainSupport = false;
            if (deviceHasNecessaryExtensions)
            {
                auto support = query_swap_chain_support(device, vk.surface);
                deviceHasSufficientSwapChainSupport = !support.formats.empty() && !support.presentModes.empty();
            }

            QueueFamilyIndices indices = get_queue_families(device, vk.surface);

            bool isSuitable = deviceHasNecessaryPropertiesAndFeatures && deviceHasNecessaryExtensions && deviceHasSufficientSwapChainSupport && indices.graphicsFamily.has_value() && indices.presentFamily.has_value();
            if (isSuitable)
            {
                vk.physicalDevice = device;
                break;
            }
        }

        if (vk.physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("No devices were suitable.");
        }

        // Create logical device
        QueueFamilyIndices indices = get_queue_families(vk.physicalDevice, vk.surface);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(), indices.presentFamily.value()
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamilyIndex : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
            queueCreateInfo.queueCount = 1;
            float queuePriority = 1.0f;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures = {};

        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = 0;
        createInfo.enabledLayerCount = enabledValidationLayers;
        createInfo.ppEnabledLayerNames = validationLayers;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(vk.physicalDevice, &createInfo, nullptr, &vk.device) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device.");
        }

        vkGetDeviceQueue(vk.device, indices.graphicsFamily.value(), 0, &vk.graphicsQueue);
        vkGetDeviceQueue(vk.device, indices.presentFamily.value(), 0, &vk.presentQueue);
    }

    auto glfw_window_extents = [window]() -> VkExtent2D
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D winExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
        return winExtent;
    };

    // Create swapchain
    auto create_swap_chain = [glfw_window_extents](VulkanContext& vk)
    {
        auto support = query_swap_chain_support(vk.physicalDevice, vk.surface);

        // choose surface format
        VkSurfaceFormatKHR surfaceFormat;
        if (support.formats.size() == 1 && support.formats[0].format == VK_FORMAT_UNDEFINED)
        {
            surfaceFormat = { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        }
        else
        {
            auto format = std::find_if(support.formats.begin(), support.formats.end(), [](const auto& f)
            {
                return f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });

            if (format != support.formats.end())
            {
                surfaceFormat = *format;
            }
            else
            {
                surfaceFormat = support.formats[0];
            }
        }

        // choose present mode
        VkPresentModeKHR presentMode;
        {
            VkPresentModeKHR best = VK_PRESENT_MODE_FIFO_KHR;
            for (const auto& p : support.presentModes)
            { 
                if (p == VK_PRESENT_MODE_MAILBOX_KHR)
                {
                    best = p;
                    break;
                }
                else if (p == VK_PRESENT_MODE_IMMEDIATE_KHR)
                {
                    best = p;
                }
            }
            presentMode = best;
        }

        // choose swap extent
        auto choose_swap_extent = [glfw_window_extents](const SwapChainSupport& support)
        {
            if (support.capabilities.currentExtent.width != 0xFFFFFFFF)
            {
                return support.capabilities.currentExtent;
            }
            else
            {
                VkExtent2D winExtent = glfw_window_extents();

                winExtent.width = std::max(support.capabilities.minImageExtent.width, std::min(support.capabilities.maxImageExtent.width, winExtent.width));
                winExtent.height = std::max(support.capabilities.minImageExtent.height, std::min(support.capabilities.maxImageExtent.height, winExtent.height));
                
                return winExtent;
            }
        };
        VkExtent2D swapExtent = choose_swap_extent(support);

        uint32_t imageCount = support.capabilities.minImageCount + 1;

        if (support.capabilities.maxImageCount > 0)
        {
            imageCount = std::min(support.capabilities.maxImageCount, imageCount);
        }

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = vk.surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = swapExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = get_queue_families(vk.physicalDevice, vk.surface);

        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(vk.device, &createInfo, nullptr, &vk.swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swap chain.");
        }

        vkGetSwapchainImagesKHR(vk.device, vk.swapChain, &imageCount, nullptr);
        vk.swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(vk.device, vk.swapChain, &imageCount, vk.swapChainImages.data());

        vk.swapChainImageFormat = surfaceFormat.format;
        vk.swapChainExtent = swapExtent;
    };
    create_swap_chain(vk);

    // Create image views
    auto create_swap_chain_image_views = [](VulkanContext& vk)
    {
        vk.swapChainImageViews.resize(vk.swapChainImages.size());

        int index = 0;
        std::for_each(vk.swapChainImages.begin(), vk.swapChainImages.end(), [&](const auto& swapChainImage)
        {
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImage;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = vk.swapChainImageFormat;

            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(vk.device, &createInfo, nullptr, &vk.swapChainImageViews[index]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image views.");
            }

            ++index;
        });
    };
    create_swap_chain_image_views(vk);

    // Create render pass
    auto create_render_pass = [](VulkanContext& vk)
    {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = vk.swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(vk.device, &renderPassInfo, nullptr, &vk.renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create render pass.");
        }
    };
    create_render_pass(vk);

    // Create graphics pipeline
    auto create_graphics_pipeline = [](VulkanContext& vk)
    {
        // Load shaders
        auto vertShaderCode = read_file("../Assets/Shaders/vert.spv");
        auto fragShaderCode = read_file("../Assets/Shaders/frag.spv");

        VkShaderModule vertShaderModule;
        {
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = vertShaderCode.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data());

            if (vkCreateShaderModule(vk.device, &createInfo, nullptr, &vertShaderModule) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create shader module.");
            }
        }

        VkShaderModule fragShaderModule;
        {
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = fragShaderCode.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data());

            if (vkCreateShaderModule(vk.device, &createInfo, nullptr, &fragShaderModule) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create shader module.");
            }
        }

        VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Configure fixed-function pipeline

        auto bindingDescription = Vertex::get_binding_description();
        auto attributeDescriptions = Vertex::get_attribute_descriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(vk.swapChainExtent.width);
        viewport.height = static_cast<float>(vk.swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = vk.swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
        };

        VkPipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 0; // change later for dynamic viewport
        dynamicState.pDynamicStates = dynamicStates;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(vk.device, &pipelineLayoutInfo, nullptr, &vk.pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pipeline layout.");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr;
        pipelineInfo.layout = vk.pipelineLayout;
        pipelineInfo.renderPass = vk.renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk.graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create graphics pipeline.");
        }

        vkDestroyShaderModule(vk.device, fragShaderModule, nullptr);
        vkDestroyShaderModule(vk.device, vertShaderModule, nullptr);
    };
    create_graphics_pipeline(vk);

    //Create frame buffers
    auto create_frame_buffers = [](VulkanContext& vk)
    {
        vk.swapChainFramebuffers.resize(vk.swapChainImageViews.size());

        int index = 0;
        std::for_each(vk.swapChainImageViews.begin(), vk.swapChainImageViews.end(), [&](const auto& swapChainImageView)
        {
            VkImageView attachments[] = {
                vk.swapChainImageViews[index]
            };

            VkFramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = vk.renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = vk.swapChainExtent.width;
            framebufferInfo.height = vk.swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(vk.device, &framebufferInfo, nullptr, &vk.swapChainFramebuffers[index]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create framebuffer.");
            }

            ++index;
        });
    };
    create_frame_buffers(vk);

    // Create command pool
    {
        QueueFamilyIndices queueFamilyIndices = get_queue_families(vk.physicalDevice, vk.surface);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        poolInfo.flags = 0;

        if (vkCreateCommandPool(vk.device, &poolInfo, nullptr, &vk.commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool.");
        }
    }

    auto create_buffer = [](const VulkanContext& vk, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(vk.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create buffer.");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(vk.device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(vk.physicalDevice, memRequirements.memoryTypeBits, properties);

        if (allocInfo.memoryTypeIndex == k_invalidMemoryType)
        {
            throw std::runtime_error("Failed to find appropriate memory type.");
        }

        if (vkAllocateMemory(vk.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate vertex buffer memory.");
        }

        vkBindBufferMemory(vk.device, buffer, bufferMemory, 0);
    };

    auto copy_buffer = [](const VulkanContext& vk, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = vk.commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(vk.device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        {
            VkBufferCopy copyRegion = {};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        }
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(vk.graphicsQueue);

        vkFreeCommandBuffers(vk.device, vk.commandPool, 1, &commandBuffer);
    };

    // Create vertex buffer
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        create_buffer(vk, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(vk.device, stagingBufferMemory, 0, bufferSize, 0, &data);
        {
            std::memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        }
        vkUnmapMemory(vk.device, stagingBufferMemory);
        
        create_buffer(vk, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vk.vertexBuffer, vk.vertexBufferMemory);

        copy_buffer(vk, stagingBuffer, vk.vertexBuffer, bufferSize);

        vkDestroyBuffer(vk.device, stagingBuffer, nullptr);
        vkFreeMemory(vk.device, stagingBufferMemory, nullptr);
    }

    // Create index buffer
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        create_buffer(vk, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(vk.device, stagingBufferMemory, 0, bufferSize, 0, &data);
        {
            std::memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        }
        vkUnmapMemory(vk.device, stagingBufferMemory);

        create_buffer(vk, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vk.indexBuffer, vk.indexBufferMemory);

        copy_buffer(vk, stagingBuffer, vk.indexBuffer, bufferSize);

        vkDestroyBuffer(vk.device, stagingBuffer, nullptr);
        vkFreeMemory(vk.device, stagingBufferMemory, nullptr);
    }

    // Create command buffers
    auto create_command_buffers = [&indices](VulkanContext& vk)
    {
        vk.commandBuffers.resize(vk.swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = vk.commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(vk.commandBuffers.size());

        if (vkAllocateCommandBuffers(vk.device, &allocInfo, vk.commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffers.");
        }

        int index = 0;
        std::for_each(vk.commandBuffers.begin(), vk.commandBuffers.end(), [&](auto& commandBuffer)
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            beginInfo.pInheritanceInfo = nullptr;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin recording command buffer.");
            }

            {
                VkRenderPassBeginInfo renderPassInfo = {};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = vk.renderPass;
                renderPassInfo.framebuffer = vk.swapChainFramebuffers[index];
                renderPassInfo.renderArea.offset = { 0, 0 };
                renderPassInfo.renderArea.extent = vk.swapChainExtent;

                VkClearValue clearColor = { 0.1f, 0.1f, 0.1f, 1.0f };
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = &clearColor;

                vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                {
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.graphicsPipeline);

                    VkBuffer vertexBuffers[] = { vk.vertexBuffer };
                    VkDeviceSize offsets[] = { 0 };
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

                    vkCmdBindIndexBuffer(commandBuffer, vk.indexBuffer, 0, VK_INDEX_TYPE_UINT16);

                    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
                }
                vkCmdEndRenderPass(commandBuffer);
            }

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command buffer.");
            }

            ++index;
        });
    };
    create_command_buffers(vk);

    // Create semaphores and fences
    const int k_maxFramesInFlight = 2;
    {
        vk.imageAvailableSemaphores.resize(k_maxFramesInFlight);
        vk.renderFinishedSemaphores.resize(k_maxFramesInFlight);
        vk.inFlightFences.resize(k_maxFramesInFlight);
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < k_maxFramesInFlight; ++i)
        {
            if (vkCreateSemaphore(vk.device, &semaphoreInfo, nullptr, &vk.imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(vk.device, &semaphoreInfo, nullptr, &vk.renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(vk.device, &fenceInfo, nullptr, &vk.inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create synchronization objects.");
            }
        }
    }

    auto cleanup_swap_chain = [](VulkanContext& vk)
    {
        for (auto frameBuffer : vk.swapChainFramebuffers)
        {
            vkDestroyFramebuffer(vk.device, frameBuffer, nullptr);
        }

        vkFreeCommandBuffers(vk.device, vk.commandPool, static_cast<uint32_t>(vk.commandBuffers.size()), vk.commandBuffers.data());

        vkDestroyPipeline(vk.device, vk.graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(vk.device, vk.pipelineLayout, nullptr);
        vkDestroyRenderPass(vk.device, vk.renderPass, nullptr);

        for (auto imageView : vk.swapChainImageViews)
        {
            vkDestroyImageView(vk.device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(vk.device, vk.swapChain, nullptr);
    };

    auto recreate_swap_chain = [&](VulkanContext& vk)
    {
        VkExtent2D extent = {};
        while (extent.width == 0 || extent.height == 0)
        {
            extent = glfw_window_extents();
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(vk.device);

        cleanup_swap_chain(vk);

        create_swap_chain(vk);
        create_swap_chain_image_views(vk);
        create_render_pass(vk);
        create_graphics_pipeline(vk);
        create_frame_buffers(vk);
        create_command_buffers(vk);
    };

    int currentFrame = 0;

    // main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // draw frame
        [&]()
        {
            vkWaitForFences(vk.device, 1, &vk.inFlightFences[currentFrame], VK_TRUE, -1);

            uint32_t imageIndex;
            VkResult result = vkAcquireNextImageKHR(vk.device, vk.swapChain, -1, vk.imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
            
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
            {
                recreate_swap_chain(vk);
                return;
            }
            else if (result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to acquire swap chain image.");
            }

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore waitSemaphores[] = { vk.imageAvailableSemaphores[currentFrame] };
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &vk.commandBuffers[imageIndex];

            VkSemaphore signalSemaphores[] = { vk.renderFinishedSemaphores[currentFrame] };
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            vkResetFences(vk.device, 1, &vk.inFlightFences[currentFrame]);

            if (vkQueueSubmit(vk.graphicsQueue, 1, &submitInfo, vk.inFlightFences[currentFrame]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit draw command buffer.");
            }

            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;

            VkSwapchainKHR swapChains[] = { vk.swapChain };
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr;

            vkQueuePresentKHR(vk.presentQueue, &presentInfo);

            vkQueueWaitIdle(vk.presentQueue);

            currentFrame = (currentFrame + 1) % k_maxFramesInFlight;
        }();
    }

    vkDeviceWaitIdle(vk.device);

    // Cleanup
    cleanup_swap_chain(vk);

    vkDestroyBuffer(vk.device, vk.vertexBuffer, nullptr);
    vkFreeMemory(vk.device, vk.vertexBufferMemory, nullptr);
    
    vkDestroyBuffer(vk.device, vk.indexBuffer, nullptr);
    vkFreeMemory(vk.device, vk.indexBufferMemory, nullptr);

    std::for_each(vk.imageAvailableSemaphores.begin(), vk.imageAvailableSemaphores.end(), [&](const auto& semaphore) { vkDestroySemaphore(vk.device, semaphore, nullptr); });
    std::for_each(vk.renderFinishedSemaphores.begin(), vk.renderFinishedSemaphores.end(), [&](const auto& semaphore) { vkDestroySemaphore(vk.device, semaphore, nullptr); });
    std::for_each(vk.inFlightFences.begin(), vk.inFlightFences.end(), [&](const auto& fence) { vkDestroyFence(vk.device, fence, nullptr); });

    vkDestroyCommandPool(vk.device, vk.commandPool, nullptr);

    vkDestroyDevice(vk.device, nullptr);

    {
        auto vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vk.vulkan, "vkDestroyDebugUtilsMessengerEXT");
        if (vkDestroyDebugUtilsMessengerEXT != nullptr)
        {
            vkDestroyDebugUtilsMessengerEXT(vk.vulkan, vk.debugMessenger, nullptr);
        }
        else
        {
            throw std::runtime_error("Failed to find vkDestroyDebugUtilsMessengerEXT");
        }
    }

    vkDestroyInstance(vk.vulkan, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cin.get();

    return 0;
}

QueueFamilyIndices get_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    auto graphicsFamilyIndex = get_index_if<VkQueueFamilyProperties>(queueFamilies, [](const auto& queueFamily, const uint32_t& index)
    {
        return queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT;
    });

    auto presentFamilyIndex = get_index_if<VkQueueFamilyProperties>(queueFamilies, [&](const auto& queueFamily, const uint32_t& index)
    {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport);
        return queueFamily.queueCount > 0 && presentSupport;
    });

    return QueueFamilyIndices{ graphicsFamilyIndex, presentFamilyIndex };
}

SwapChainSupport query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupport ret;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &ret.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount > 0)
    {
        ret.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, ret.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount > 0)
    {
        ret.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, ret.presentModes.data());
    }

    return ret;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    fprintf(stderr, "Validation layer: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

uint32_t find_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    return k_invalidMemoryType;
}

static std::vector<char> read_file(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file.");
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}