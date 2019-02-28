#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <cstdio>

#include <vector>
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

SwapChainSupport query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface);

void on_key_event(GLFWwindow* window, int key, int scancode, int action, int mods);
void on_key_press(GLFWwindow* window, int key, int scancode, int mods, bool repeat);
void on_key_release(GLFWwindow* window, int key, int scancode, int mods);

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

int main()
{
#if 0
    test_everything_works();
    return 0;
#endif

    GLFWwindow* window = nullptr;
    const int SCREEN_WIDTH = 1280, SCREEN_HEIGHT = 720;
    
    std::vector<const char*> enabledExtensions;

    const uint32_t validationLayerCount = 1;
    const char* validationLayers[validationLayerCount]{
        "VK_LAYER_LUNARG_standard_validation",
    };

    const bool enableValidation = true;
    uint32_t enabledValidationLayers = (enableValidation) ? validationLayerCount : 0;

    // Init GLFW and window
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "VulkanTutorial", nullptr, nullptr);
        glfwSetKeyCallback(window, on_key_event);

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        enabledExtensions.insert(enabledExtensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Init Vulkan
    VkInstance vulkan;
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

        if (vkCreateInstance(&createInfo, nullptr, &vulkan) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan instance.");
            return 1;
        }
    }

    // Create window surface
    VkSurfaceKHR surface;
    {
        if (glfwCreateWindowSurface(vulkan, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw new std::runtime_error("Failed to create window surface.");
            return 1;
        }
    }

    // Find physical device
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    {
        // get all physical devices
        std::uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(vulkan, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support.");
            return 1;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(vulkan, &deviceCount, devices.data());

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
                auto support = query_swap_chain_support(device, surface);
                deviceHasSufficientSwapChainSupport = !support.formats.empty() && !support.presentModes.empty();
            }

            QueueFamilyIndices indices = get_queue_families(device, surface);

            bool isSuitable = deviceHasNecessaryPropertiesAndFeatures && deviceHasNecessaryExtensions && deviceHasSufficientSwapChainSupport && indices.graphicsFamily.has_value() && indices.presentFamily.has_value();
            if (isSuitable)
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("No devices were suitable.");
            return 1;
        }

        // Create logical device
        QueueFamilyIndices indices = get_queue_families(physicalDevice, surface);

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

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device.");
            return 1;
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Create the swapchain
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    {
        auto support = query_swap_chain_support(physicalDevice, surface);

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
        VkExtent2D swapExtent;
        {
            if (support.capabilities.currentExtent.width != 0xFFFFFFFF)
            {
                swapExtent = support.capabilities.currentExtent;
            }
            else
            {
                VkExtent2D winExtent = { SCREEN_WIDTH, SCREEN_HEIGHT };
                winExtent.width = std::max(support.capabilities.minImageExtent.width, std::min(support.capabilities.maxImageExtent.width, winExtent.width));
                winExtent.height = std::max(support.capabilities.minImageExtent.height, std::min(support.capabilities.maxImageExtent.height, winExtent.height));
                swapExtent = winExtent;
            }
        }

        uint32_t imageCount = support.capabilities.minImageCount + 1;

        if (support.capabilities.maxImageCount > 0)
        {
            imageCount = std::min(support.capabilities.maxImageCount, imageCount);
        }

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = swapExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = get_queue_families(physicalDevice, surface);

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

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swap chain.");
            return -1;
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = swapExtent;
    }

    // Create image views
    std::vector<VkImageView> swapChainImageViews;
    {
        swapChainImageViews.resize(swapChainImages.size());

        int index = 0;
        std::for_each(swapChainImages.begin(), swapChainImages.end(), [&](const auto& swapChainImage)
        {
            VkImageViewCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImage;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[index]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image views.");
            }

            index += 1;
        });
    }

    // Create graphics pipeline
    {

    }

    // Setup debug messenger
    VkDebugUtilsMessengerEXT debugMessenger;
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = vk_debug;
        createInfo.pUserData = nullptr;

        auto vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vulkan, "vkCreateDebugUtilsMessengerEXT");

        if (vkCreateDebugUtilsMessengerEXT != nullptr)
        {
            vkCreateDebugUtilsMessengerEXT(vulkan, &createInfo, nullptr, &debugMessenger);
        }
        else
        {
            throw std::runtime_error("Failed to find vkCreateDebugUtilsMessengerEXT");
            return 1;
        }
    }

    // main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }

    // Cleanup

    for (auto imageView : swapChainImageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }
    
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    vkDestroyDevice(device, nullptr);

    {
        auto vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vulkan, "vkDestroyDebugUtilsMessengerEXT");
        if (vkDestroyDebugUtilsMessengerEXT != nullptr)
        {
            vkDestroyDebugUtilsMessengerEXT(vulkan, debugMessenger, nullptr);
        }
        else
        {
            throw std::runtime_error("Failed to find vkDestroyDebugUtilsMessengerEXT");
            return 1;
        }
    }

    vkDestroyInstance(vulkan, nullptr);
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

void on_key_event(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    switch (action)
    {
    default:
    case GLFW_PRESS: on_key_press(window, key, scancode, action, false); return;
    case GLFW_REPEAT: on_key_press(window, key, scancode, action, true); return;
    case GLFW_RELEASE: on_key_release(window, key, scancode, action); return;
    }
}
void on_key_press(GLFWwindow* window, int key, int scancode, int mods, bool repeat)
{
    if (key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void on_key_release(GLFWwindow* window, int key, int scancode, int mods)
{
    // nothing
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    fprintf(stderr, "Validation layer: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
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